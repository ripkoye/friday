# AudioReceiver_async_print.py
import asyncio
import json
import struct
import time
from typing import Tuple, Optional, List

import os

import numpy as np

import torch
from faster_whisper import WhisperModel

import redis

# --------------------------- Config ---------------------------

HOST = "0.0.0.0"
PORT = 8080

SAMPLE_RATE = 16000
DTYPE = np.int16
FLOAT_SCALE = 32768.0

QUEUE_MAXSIZE = 200

MIN_SECONDS_TO_TRANSCRIBE = 0.4
VAD_INCLUDE_THRESHOLD = None # None to disable

REDIS_HOST= "127.0.0.1"
REDIS_PORT = 6379

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    db=0,
    password="98679Ryna71@"
)

TRANSCRIBE_OPTS = dict(
    language="en",
    beam_size=1,
    best_of=1,
    word_timestamps=False,
    temperature=0.0,
    vad_filter=True,
    condition_on_previous_text=False,
)

MODEL_NAME = "distil-small.en"
USE_CUDA = torch.cuda.is_available()
COMPUTE_TYPE = "float16" if USE_CUDA else "int8"
CPU_THREADS = max(1, torch.get_num_threads())



# --------------------------------------------------------------



# --------------------------------------------------------------

def now_s() -> float:
    return time.time()

class Transcriber:
    def __init__(self):
        print(f"[Init] Loading Faster-Whisper model {MODEL_NAME} on "
              f"{'cuda' if USE_CUDA else 'cpu'} ({COMPUTE_TYPE}) ...")
        self.model = WhisperModel(
            MODEL_NAME,
            device="cuda" if USE_CUDA else "cpu",
            compute_type=COMPUTE_TYPE,
            cpu_threads=CPU_THREADS
        )

    def pcm16_to_float32(self, pcm: bytes) -> np.ndarray:
        arr = np.frombuffer(pcm, dtype=DTYPE).astype(np.float32) / FLOAT_SCALE
        if arr.size == 0:
            return arr
        return np.clip(arr, -1.0, 1.0)

    def concat_float32(self, chunks: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(chunks, dtype=np.float32) if chunks else np.array([], dtype=np.float32)

# Global
audio_queue: "asyncio.Queue[Tuple[bytes, float, Optional[float]]]" = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
transcriber = Transcriber()

# ------------------------- Networking -------------------------

async def read_exactly(reader: asyncio.StreamReader, n: int) -> bytes:
    return await reader.readexactly(n)

async def recv_message(reader: asyncio.StreamReader) -> Tuple[dict, bytes]:
    raw_len = await read_exactly(reader, 4)
    (header_len,) = struct.unpack(">I", raw_len)
    header_bytes = await read_exactly(reader, header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    audio_len = int(header["length"])
    audio_bytes = await read_exactly(reader, audio_len)
    return header, audio_bytes

def prepare_audio_for_silero(audio_bytes: bytes, stereo=False) -> torch.Tensor:
    x = np.frombuffer(audio_bytes, dtype=np.int16)
    if x.size == 0:
        return torch.zeros(1, dtype=torch.float32)

    # If capture might be stereo, downmix to mono
    if stereo and x.size % 2 == 0:
        x = x.reshape(-1, 2).astype(np.int32)
        x = ((x[:, 0] + x[:, 1]) // 2).astype(np.int16)

    # int16 -> float32 in [-1, 1]
    x = x.astype(np.float32) / 32768.0
    np.clip(x, -1.0, 1.0, out=x)

    # Small DC guard helps some setups
    if x.size >= 160:         # ~10 ms
        x -= np.mean(x)

    # Silero VAD requires exactly 512 samples for 16kHz
    required_samples = 512
    if x.size >= required_samples:
        x = x[:required_samples]  # Take first 512 samples
    else:
        # Pad with zeros if too short
        x = np.pad(x, (0, required_samples - x.size))

    return torch.from_numpy(np.ascontiguousarray(x))  # 1-D, CPU, float32


def silero_prob(model, audio_bytes: bytes) -> float | None:
    x = prepare_audio_for_silero(audio_bytes, stereo=False)
    try:
        with torch.no_grad():
            p = model(x, 16000)
        p = float(p.item() if hasattr(p, "item") else p)
        return 0.0 if not (0.0 <= p <= 1.0) else p
    except Exception as e:
        print(f"[Silero ERR] {type(e).__name__}: {e} | shape={tuple(x.shape)} dtype={x.dtype} "
              f"min={float(x.min()) if x.numel() else 'NA'} max={float(x.max()) if x.numel() else 'NA'}")
        return None


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, silero_model) -> None:
    peer = writer.get_extra_info("peername")
    print(f"[Conn] Client connected: {peer}")
    try:
        while True:
            header, audio_bytes = await recv_message(reader)
            ts = float(header.get("timestamp", now_s()))
            try:
                sp = silero_prob(silero_model, audio_bytes)
            except Exception:
                sp = None
            if VAD_INCLUDE_THRESHOLD is None or (sp is not None and sp >= VAD_INCLUDE_THRESHOLD):
                try:
                    await audio_queue.put((audio_bytes, ts, sp))
                except asyncio.QueueFull:
                    print("[Warn] audio_queue full; dropping chunk")
    except (asyncio.IncompleteReadError, ConnectionResetError):
        pass
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"[Conn] Client disconnected: {peer}")

# ----------------------- Consumer Task ------------------------

async def consumer_task() -> None:
    # --------- VAD / segmentation params (tune to taste) ----------
    VAD_SPEECH_ON      = 0.20  # enter speech when prob >= this
    VAD_SPEECH_OFF     = 0.10  # exit speech when prob <= this
    MIN_SILENCE_SEC    = 1.00  # consecutive silence needed to flush (longer gap)
    MAX_UTTERANCE_SEC  = 30.0  # safety cap (prevents holding forever)
    IDLE_FLUSH_SEC     = 0.50  # no chunks while "in speech" -> flush
    PRE_ROLL_SEC       = 0.90  # prepend a little pre-speech audio
    # -------------------------------------------------------------

    loop = asyncio.get_running_loop()

    state = "silence"   # or "speech"
    utter_chunks: List[np.ndarray] = []
    pre_roll: List[np.ndarray] = []
    pre_roll_samples_target = int(PRE_ROLL_SEC * SAMPLE_RATE)

    silence_accum = 0.0
    last_chunk_wall = now_s()
    utter_start_ts: Optional[float] = None
    utter_samples = 0
    



    def chunk_seconds(n_bytes: int) -> float:
        return n_bytes / (2.0 * SAMPLE_RATE)  # int16 mono: 2 bytes/sample

    def seconds_in_chunks(chunks: List[np.ndarray]) -> float:
        return sum(c.size for c in chunks) / float(SAMPLE_RATE) if chunks else 0.0

    async def flush_if_any(reason: str):
        nonlocal state, utter_chunks, utter_samples, utter_start_ts, silence_accum
        dur_sec = seconds_in_chunks(utter_chunks)
        if dur_sec <= 0.0:
            # reset state
            state = "silence"
            silence_accum = 0.0
            utter_chunks.clear()
            utter_samples = 0
            utter_start_ts = None
            return

        audio_array = transcriber.concat_float32(utter_chunks)
        # reset buffers *before* transcribe so intake isn't blocked on errors
        state = "silence"
        silence_accum = 0.0
        utter_chunks.clear()
        utter_samples = 0

        if dur_sec < MIN_SECONDS_TO_TRANSCRIBE:
            utter_start_ts = None
            return

        print(f"[FLUSH:{reason}] {dur_sec:.2f}s audio (start_ts={utter_start_ts})")

        try:
            started = now_s()
            segments, info = await loop.run_in_executor(
                None,
                lambda: transcriber.model.transcribe(audio_array, **TRANSCRIBE_OPTS)
            )
            segments_list = list(segments)
            print(f"[XR] segs={len(segments_list)} took={now_s()-started:.2f}s utt={dur_sec:.2f}s")
        except Exception as e:
            print(f"[ERR] transcribe failed: {e}")
            utter_start_ts = None
            return

        # Write to Redis with wall-clock offsets if we have them (non-blocking)
        async def write_to_redis():
            for seg in segments_list:
                try:
                    if utter_start_ts is not None:
                        start_abs = utter_start_ts + float(seg.start or 0.0)
                        end_abs   = utter_start_ts + float(seg.end   or 0.0)
                        fields = {"start": f"{start_abs:.2f}", "end": f"{end_abs:.2f}", "text": seg.text}
                    else:
                        fields = {"start": f"{float(seg.start or 0.0):.2f}",
                                  "end":   f"{float(seg.end   or 0.0):.2f}",
                                  "text":  seg.text}
                    r.xadd("transcript:session1", fields, maxlen=10000, approximate=True)
                    print(f"[OUT] [{fields['start']}→{fields['end']}] {seg.text}")
                except Exception as e:
                    print(f"[ERR] xadd failed: {e}")
        
        # Fire and forget Redis writes to avoid blocking
        asyncio.create_task(write_to_redis())
        utter_start_ts = None

    # ----------- main loop -----------
    while True:
        try:
            try:
                audio_bytes, ts, sp = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                last_chunk_wall = now_s()
            except asyncio.TimeoutError:
                # If we're "in speech" but no chunks are arriving, treat as a pause
                if state == "speech" and (now_s() - last_chunk_wall) >= IDLE_FLUSH_SEC:
                    await flush_if_any("idle")
                continue

            dur = chunk_seconds(len(audio_bytes))
            chunk_float = transcriber.pcm16_to_float32(audio_bytes)
            
            # DEBUG: Print every VAD result
            sp_str = f"{sp:.3f}" if sp is not None else "None"
            print(f"[VAD] prob={sp_str} state={state} silence_accum={silence_accum:.2f}s dur={dur:.3f}s", end="")

            if state == "silence":
                # Maintain pre-roll buffer
                pre_roll.append(chunk_float)
                total_samples = sum(c.size for c in pre_roll)
                while total_samples > pre_roll_samples_target and pre_roll:
                    drop = pre_roll.pop(0)
                    total_samples -= drop.size

                # Enter speech?
                if sp is not None and sp >= VAD_SPEECH_ON:
                    print(f" → ENTER_SPEECH!")
                    state = "speech"
                    silence_accum = 0.0
                    utter_start_ts = ts  # wall-clock
                    utter_chunks = pre_roll[:] + [chunk_float]
                    utter_samples = sum(c.size for c in utter_chunks)
                    pre_roll.clear()
                else:
                    print(f" → stay_silence")
                    continue

            else:  # state == "speech"
                utter_chunks.append(chunk_float)
                utter_samples += chunk_float.size

                # Hysteresis on/off to avoid chatter
                if sp is None:
                    print(f" → no_vad")
                    silence_accum = 0.0  # conservative: assume still speech
                elif sp <= VAD_SPEECH_OFF:
                    silence_accum += dur
                    print(f" → silence_detected")
                else:
                    silence_accum = 0.0
                    print(f" → speech_continues")

                if silence_accum >= MIN_SILENCE_SEC:
                    print(f" → FLUSH_SILENCE_GAP!")
                    await flush_if_any("silence-gap")
                    continue

                if (utter_samples / float(SAMPLE_RATE)) >= MAX_UTTERANCE_SEC:
                    print(f" → FLUSH_MAX_UTTERANCE!")
                    await flush_if_any("max-utterance")
                    continue

        except Exception as e:
            print(f"[FATAL] consumer loop crashed: {e}")
            await asyncio.sleep(0.5)


# --------------------------- Main -----------------------------

async def main():
    print("[Init] Loading Silero VAD ...")
    silero_model, utils = torch.hub.load(
        repo_or_dir="/home/ripkoye/friday/silero-vad-master", model="silero_vad", source="local", force_reload=False
    )

    asyncio.create_task(consumer_task())

    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, silero_model),
        host=HOST, port=PORT, start_serving=True
    )
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(f"[Server] Listening on {addrs}; CUDA={'yes' if USE_CUDA else 'no'}")

    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Exit] Shutting down.")
