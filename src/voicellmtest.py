# client_basic.py
import os, json
from ollama import Client

# If the model runs on another machine:
#os.environ["OLLAMA_HOST"] = "http://SERVER_IP:11434"
client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

MODEL = "qwen2.5"  # whatever name you used in `ollama create`

messages=[
    {"role":"system","content":"You are helpful."},
    {"role":"user","content":"Give me a two-sentence intro to LLMs."}
]

# 1) One-shot chat
resp = client.chat(model=MODEL, messages=messages, options={"temperature":0.3, "num_ctx":8192}, keep_alive="5m")
print(resp["message"]["content"])

# 2) Streaming (token-by-token)
stream = client.chat(model=MODEL, messages=[
    {"role":"user","content":"Explain KV cache briefly."}
], stream=True, options={"temperature":0.3}, keep_alive="5m")

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
print()
