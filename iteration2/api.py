from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from pydantic import BaseModel
import torch
import uvicorn

BERT_MODEL_NAME = "/home/ripkoye/friday/models/friday_agent_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERTDIM = 768

bert1Tokenizer = None
bert1Model = None
bert2Tokenizer = None
bert2Model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    bert1Initialization()
    bert2Initialization()
    qwenInitialization()
    yield
    # Shutdown (if needed)

app = FastAPI(lifespan=lifespan)

@app.post("/bert1embed")
async def embed1(req: Request):
    body = await req.json()
    texts = body.get("texts", [])
    inputs = bert1Tokenizer(texts, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}  # move to GPU/CPU
    with torch.no_grad():
        outputs = bert1Model(**inputs)
    return {"predictions": outputs.logits.argmax(-1).cpu().tolist()}

def bert1Initialization():
    global bert1Tokenizer, bert1Model
    bert1Tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert1Model = AutoModelForSequenceClassification.from_pretrained(
            "/home/ripkoye/friday/models/friday_agent_model"
        ).to(DEVICE)

#2 embed is regular bert
@app.post("/bert2embed")
async def embed2(req: Request):  # Fix function name
    body = await req.json()
    texts = body.get("texts")
    inputs = bert2Tokenizer(texts, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert2Model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Pool embeddings
    return {"embeddings": embeddings.cpu().tolist()}

def bert2Initialization():
    global bert2Tokenizer, bert2Model
    from transformers import AutoModel  # Import base model
    bert2Tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert2Model = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)

#-------qwen
from ollama import Client
import os
qwenMODEL = "qwen2.5"
client = None

def qwenInitialization():
    global client
    client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

@app.post("/qwenCall")
async def qwenCall(req: Request) -> str:
    try:
        body = await req.json()
        history = body.get("history", [])
        
        output = client.chat(
            model=qwenMODEL, messages=history, stream=False, 
            keep_alive="5m", options={"temperature":0.3, 
            "stop": ["<|im_end|>", "<|im_start|>"]}
            )
        response = output["message"]["content"]
        return response
    except Exception as e:
        return f"Error:{e}"


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
