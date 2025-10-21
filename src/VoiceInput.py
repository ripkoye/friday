# AudioReceiver.py
import socket
import json
import numpy as np
import torch, torchaudio
import asyncio
import os
import time
from faster_whisper import WhisperModel

import struct

def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return bytes(buf)

def recv_message(sock):
    raw_len = recv_exact(sock, 4)
    (header_len,) = struct.unpack(">I", raw_len)
    header_bytes = recv_exact(sock, header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    audio_bytes = recv_exact(sock, int(header["length"]))
    return header, audio_bytes



class AudioBuffer:
    def __init__(self, utils, model):
        self.audio_buffer = []
        self.whisper_model = WhisperModel("small", device="cuda")
        self.transcription_file = "supportcache/current_transcription.txt"
        self.utils = utils
        self.model = model
        self.last_processing_time = time.time()
        self.PROCESSING_INTERVAL = 5.0  # Process every 5 seconds

    async def add(self, audio_segment):
        audio_bytes, timestamp, prob = audio_segment
        
        # Always add to buffer (we'll let Silero VAD determine speech boundaries later)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer.append((audio_array, timestamp))
        
        # Process every 5 seconds regardless of chunk count
        current_time = time.time()
        if current_time - self.last_processing_time >= self.PROCESSING_INTERVAL:
            print(f"TIME TO PROCESS! Processing {len(self.audio_buffer)} chunks after {self.PROCESSING_INTERVAL}s...")
            await self.process_speech_segment()
            self.audio_buffer = []
            self.last_processing_time = current_time
        
        return None
    
    async def process_speech_segment(self):
        """Process accumulated speech buffer with Whisper"""
        if len(self.audio_buffer) == 0:
            return
        
        print(f"PROCESSING SPEECH SEGMENT! Buffer size: {len(self.audio_buffer)}")
        
        # Combine all numpy arrays into one stream
        combined_arrays = [chunk[0] for chunk in self.audio_buffer]
        audio_stream = np.concatenate(combined_arrays)
        
        # Ensure audio is in the right format for Whisper
        if audio_stream.dtype != np.float32:
            audio_stream = audio_stream.astype(np.float32)
        
        # Ensure audio is in the right range (-1 to 1)
        if audio_stream.max() > 1.0 or audio_stream.min() < -1.0:
            audio_stream = np.clip(audio_stream, -1.0, 1.0)
        
        print(f"Combined audio stream length: {len(audio_stream)} samples, dtype: {audio_stream.dtype}")
        
        loop = asyncio.get_running_loop()

        try:
            # Transcribe with Whisper (no need for Silero VAD again)
            print("Starting Whisper transcription...")

            segments, info = await loop.run_in_executor(
                None,
                lambda: self.whisper_model.transcribe(audio_stream, vad_filter=True)
            )
            
            # Convert generator to list to get length
            segments_list = list(segments)
            print(f"Whisper returned {len(segments_list)} segments")
            
            # Write transcription to file
            for segment in segments_list:
                print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()

async def start_audio_server():
    """Minimal server to receive audio"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 8080))  # Use port 8080 instead of 2222
    sock.listen(1)
    
    print("Audio server started on port 8080")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available")
    else:
        print("CUDA is not available")

    model, utils = torch.hub.load(repo_or_dir='./silero-vad-master', model='silero_vad',source = 'local', force_reload=True)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    AudioProcessor = AudioBuffer(utils, model)

    while True:
        client, addr = sock.accept()
        print(f"Client connected from {addr}")
        
        try:
            while True:
                # fead header
                header_data, audio_bytes = recv_message(client)
                if not header_data:
                    break
                
                # header_data is already a dictionary, no need to parse
                header = header_data
                
                # audio_bytes is already the complete audio data
                # No need to read more or process remaining audio
                
                #print(f"Received {len(audio_bytes)} bytes")

                speech_prob = silero(audio_bytes, model)

                await AudioProcessor.add([audio_bytes, header['timestamp'], speech_prob])
    
        except Exception as e:
            print(f"Error: {e}")
        finally:
            client.close()

def silero(audio_bytes, model):
    audio_tensor = prepare_audio_for_silero(audio_bytes)

    speech_prob = model(audio_tensor,16000).item()

    #print(f"Speech Probability: {speech_prob}")
    return speech_prob

def prepare_audio_for_silero(audio_bytes):
    audio_numpy = np.frombuffer(audio_bytes, dtype=np.int16)
    
    # Reshape to 512-sample chunks for Silero VAD (16kHz)
    chunk_size = 512
    if len(audio_numpy) >= chunk_size:
        # Take the first 512 samples
        audio_numpy = audio_numpy[:chunk_size]
    else:
        # Pad with zeros if too short
        audio_numpy = np.pad(audio_numpy, (0, chunk_size - len(audio_numpy)))
    
    # .copy() fixes the writable tensor warning
    audio_tensor = torch.from_numpy(audio_numpy.copy())
    audio_tensor = audio_tensor.float()
    return audio_tensor

if __name__ == "__main__":
    asyncio.run(start_audio_server())
