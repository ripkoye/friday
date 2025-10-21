import redis
import json
import asyncio
import socket
import struct
import numpy as np
from kokoro import KPipeline

class TTSServer:
    def __init__(self):
        self.redis_client = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True, password='98679Ryna71@')
        self.pipeline = KPipeline(lang_code="b")
        self.client_writer = None
        
    async def listen_for_tts_requests(self):
        """Listen for TTS requests from VoiceAgent"""
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe("tts_requests")
        
        print("[TTSServer] Listening for TTS requests...")
        
        try:
            while True:
                # Use get_message with timeout to make it non-blocking
                message = pubsub.get_message(timeout=0.1)
                if message is None:
                    await asyncio.sleep(0.1)  # Yield control to other tasks
                    continue
                    
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        text = data.get('text', '')
                        
                        print(f"[TTSServer] Generating TTS for: '{text[:50]}...'")
                        
                        # Generate audio with timeout and better error handling
                        try:
                            audio_chunks = []
                            print("[TTSServer] Starting Kokoro pipeline...")
                            
                            for i, (_, _, audio) in enumerate(self.pipeline(text, voice="bm_george")):
                                print(f"[TTSServer] Got chunk {i}, size: {len(audio) if hasattr(audio, '__len__') else 'unknown'}")
                                audio_chunks.append(audio)
                            
                            print(f"[TTSServer] Generated {len(audio_chunks)} audio chunks")
                            
                            if audio_chunks:
                                full_audio = np.concatenate(audio_chunks)
                                print(f"[TTSServer] Final audio shape: {full_audio.shape}")
                                await self.send_audio_to_client(full_audio)
                            else:
                                print("[TTSServer] No audio chunks generated!")
                                
                        except Exception as tts_error:
                            print(f"[TTSServer] TTS generation failed: {tts_error}")
                            print(f"[TTSServer] Error type: {type(tts_error)}")
                            import traceback
                            traceback.print_exc()
                            
                            # Send a simple beep or skip TTS for now
                            print("[TTSServer] Skipping TTS due to error")
                            
                    except Exception as e:
                        print(f"[TTSServer] Error processing TTS request: {e}")
                        
        except KeyboardInterrupt:
            print("\n[TTSServer] Shutting down...")
            pubsub.close()
    
    async def send_audio_to_client(self, audio_data):
        """Send audio data to connected client using framed protocol"""
        if self.client_writer:
            try:
                # Convert to 16-bit PCM for compatibility
                audio_int16 = (audio_data * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                
                # Create header matching Client expectations
                header_obj = {
                    "type": "tts_audio",
                    "length": len(audio_bytes),
                    "sample_rate": 24000,
                    "channels": 1,
                    "sample_width": 2,  # 16-bit = 2 bytes
                    "timestamp": time.time()
                }
                
                # Pack frame: [4-byte header length][JSON header][audio payload]
                header_json = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
                frame = struct.pack(">I", len(header_json)) + header_json + audio_bytes
                
                self.client_writer.write(frame)
                await self.client_writer.drain()
                print(f"[TTSServer] Sent {len(audio_bytes)} bytes to client")
                
            except Exception as e:
                print(f"[TTSServer] Error sending audio: {e}")
                self.client_writer = None
    
    async def handle_client_connections(self):
        """Handle TCP connections from clients"""
        print("[TTSServer] Starting TCP server on port 8081...")
        
        try:
            server = await asyncio.start_server(
                self.handle_client, 
                host='0.0.0.0', 
                port=8081
            )
            
            print("[TTSServer] TCP server listening on port 8081")
            
            async with server:
                await server.serve_forever()
                
        except Exception as e:
            print(f"[TTSServer] Failed to start TCP server: {e}")
            import traceback
            traceback.print_exc()
    
    async def handle_client(self, reader, writer):
        """Handle individual client connection"""
        addr = writer.get_extra_info('peername')
        print(f"[TTSServer] Client connected from {addr}")
        
        self.client_writer = writer
        
        # Keep connection alive
        try:
            while True:
                await asyncio.sleep(1)
        except Exception as e:
            print(f"[TTSServer] Client {addr} disconnected: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            self.client_writer = None

async def main():
    tts_server = TTSServer()
    
    print("[TTSServer] Starting both Redis listener and TCP server...")
    
    # Run both TTS listener and TCP server
    try:
        await asyncio.gather(
            tts_server.listen_for_tts_requests(),
            tts_server.handle_client_connections()
        )
    except Exception as e:
        print(f"[TTSServer] Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import time
    asyncio.run(main())
