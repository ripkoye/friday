# AudioReceiver_async_print.py
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import struct
import time
from typing import Tuple, Optional, List

# Suppress torchaudio deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

import numpy as np

import torch
from faster_whisper import WhisperModel

import redis

# Add diart imports
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.models import SegmentationModel, EmbeddingModel

# Add soundfile for audio file handling
import soundfile as sf

# Add tempfile for temporary file handling
import tempfile

# Add ThreadPoolExecutor for worker threads
from concurrent.futures import ThreadPoolExecutor

# Add HuggingFace imports
from huggingface_hub import HfFolder

# --------------------------- Config ---------------------------

HOST = "0.0.0.0"
PORT = 8080

SAMPLE_RATE = 16000
DTYPE = np.int16
FLOAT_SCALE = 32768.0

QUEUE_MAXSIZE = 200

MIN_SECONDS_TO_TRANSCRIBE = 0.4
VAD_INCLUDE_THRESHOLD = None # None to disable

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    db=0,
    password=os.getenv("REDIS_PASSWORD", "")
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

def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)

class DiartRuntime:
    def __init__(self):
        print("[Init] Loading Diart models...")

        # --- auth (env or CLI login) ---
        token = (os.getenv("HUGGINGFACE_TOKEN")
                 or os.getenv("HF_TOKEN")
                 or os.getenv("HUGGINGFACE_HUB_TOKEN")
                 or HfFolder.get_token())
        self.token = token

        # --- diart configuration ---
        # Use default diart configuration which handles model loading internally
        self.pipeline = SpeakerDiarization()
        print("[Init] Diart pipeline loaded")

        # thresholds for speaker verification
        self.you_threshold = 0.82   # cosine threshold to call a segment "ryan"
        self.vote_ratio    = 0.50   # majority of diarized speech must be Ryan
        
        print(f"[Init] Thresholds: you_threshold={self.you_threshold}, vote_ratio={self.vote_ratio}")

        # --- enrollment ---
        self.reference_embeddings = {}
        self.load_reference_embeddings()

        print("[Init] Diart models loaded successfully")

    def load_reference_embeddings(self):
        """Load single-user enrollment vector; normalize once."""
        npz_path = "/home/ripkoye/friday/supportcache/enrollment_v1.npz"
        data = np.load(npz_path, allow_pickle=False)
        if "embedding" not in data:
            raise ValueError("Enrollment NPZ must contain an 'embedding' key.")
        emb = data["embedding"].astype(np.float32)
        if emb.ndim != 1:
            raise ValueError(f"Unexpected embedding shape: {emb.shape}")
        self.reference_embeddings["ryan"] = _l2(emb)
        print(f"[Init] Loaded ryan embedding with shape: {emb.shape}")

    # --- REAL speaker verification using diart + cosine ---
    def label_chunk(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """
        Use diart to diarize the utterance, extract embeddings for each speaker,
        compare to enrollment; majority vote over speech duration.
        Returns: "ryan" or "other"
        """
        if self.pipeline is None or not self.reference_embeddings:
            return "other"

        # Write a tiny temp WAV for robust backend behavior
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_array, sample_rate, subtype="PCM_16")
                tmp_path = tmp.name

            # 1) Use diart to diarize and get speaker embeddings
            diarization = self.pipeline(tmp_path)
            
            # 2) Extract embeddings for each speaker segment
            ryan_vec = self.reference_embeddings["ryan"]
            total_speech = 0.0
            ryan_speech  = 0.0

            # Process each speaker segment from diart output
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_time = float(turn.start)
                end_time = float(turn.end)
                dur = max(0.0, end_time - start_time)
                if dur <= 0.0:
                    continue

                # Extract embedding for this segment using diart's embedding model
                embedding = self.pipeline.config.embedding({"audio": tmp_path, "start": start_time, "end": end_time})
                
                if embedding is not None:
                    # Normalize embedding
                    embedding = _l2(embedding.astype(np.float32))
                    score = float(np.dot(embedding, ryan_vec))  # cosine similarity

                    total_speech += dur
                    if score >= self.you_threshold:
                        ryan_speech += dur
                        print(f"[DEBUG] → RYAN (score {score:.3f} >= {self.you_threshold})")
                    else:
                        print(f"[DEBUG] → OTHER (score {score:.3f} < {self.you_threshold})")

            if total_speech <= 0.0:
                return "other"

            you_ratio = ryan_speech / total_speech
            result = "ryan" if you_ratio >= self.vote_ratio else "other"
            print(f"[DEBUG] Final decision: {result} (ratio={you_ratio:.3f}, threshold={self.vote_ratio})")
            return result

        except Exception as e:
            print(f"[Error] Diart diarization/embedding extraction failed: {e}")
            return "other"
        finally:
            if tmp_path:
                try: os.unlink(tmp_path)
                except Exception: pass


# Global
audio_queue: "asyncio.Queue[Tuple[bytes, float, Optional[float]]]" = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
transcriber = Transcriber()
diart_runtime = DiartRuntime() # Initialize DiartRuntime

# ThreadPoolExecutor for CPU-intensive tasks
thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="AudioWorker")
