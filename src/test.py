from kokoro import KPipeline
import numpy as np

pipeline = KPipeline(lang_code="a")  # uses cached weights
text = "Kokoro is an open-weight TTS model."

audio_chunks = []
for _, _, audio in pipeline(text, voice="af_heart"):
    audio_chunks.append(audio)
    full = np.concatenate(audio_chunks)
