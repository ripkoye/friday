import redis
import torch
import asyncio, time, json
from datetime import datetime, timezone
import numpy as np
import requests, re

REDIS_HOST= "127.0.0.1"
REDIS_PORT = 6379
STREAM = "transcript:session1"

#----Connection
r = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True, password='98679Ryna71@')
rb = redis.Redis(host="127.0.0.1", port=6379, decode_responses=False, password='98679Ryna71@')

#----Embedding Utils

def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def to_bytes(v: np.ndarray) -> bytes:
    return v.astype(np.float32, copy=False).tobytes()

async def getRedis():
    last_id = "0"  # Start from beginning
    string = ""  # Initialize string outside the loop
    while True:
        resp = r.xread({STREAM: last_id}, block=5000, count=10)
        if not resp:  # No messages
            await asyncio.sleep(0.1)
            continue
            
        for stream, messages in resp:
            for msg_id, fields in messages:
                text = fields['text']
                start = fields['start']
                end = fields['end']
                string += text  # Accumulate text
                last_id = msg_id  # Update to continue from this message
                print(stream, msg_id, fields)
                
        if string and "friday" in string.lower():  # Only process if we have text and it contains "friday"
            pred = await bertClassifier(string)
            asyncio.create_task(classifiers(string))
            r.delete("transcript:session1")
            string = ""  # Reset string after processing

async def bertClassifier(text):
    try:
        response = requests.post(
            "http://localhost:8000/bert1embed",
            json={"texts": [text]}  # Fix: needs to be a list
        )
        predictions = response.json()["predictions"]
        return predictions[0] if predictions else 0  # Return first prediction or 0
    except Exception as e:
        print(f"BERT classifier error: {e}")
        return 0



async def classifiers(text):
    """classifier 1 & 2 & 3"""
    [
        2,                        # total matches
        "stmem:fact:42",          # key of first doc
        ["$.text", "hello redis"],# fields
        "stmem:fact:84",          # key of second doc
        ["$.text", "semantic search rocks"]
    ]

    try:
        response = requests.post(
            "http://localhost:8000/bert2embed",
            json={"texts": [text]}  # Fix: needs to be a list
        )
        embeddings = response.json()["embeddings"]
        embedding_vector = np.array(embeddings[0])
        resp2 = await classifier2(text, embedding_vector)
        resp3 = await classifier3(text, embedding_vector)
        
        content2, content3 = await format_llm(resp2, resp3)

        r.publish("voice_agent_trigger", json.dumps({  # Fix: json.dumps not json.dump
            "message": text,
            "resp2": content2,
            "resp3": content3
        }))
    except Exception as e:
        print(f"Classifiers error: {e}")
        r.publish("voice_agent_trigger", json.dumps({
            "message": text,
            "error": str(e)
        }))


async def format_llm(list2, list3):
    async def resp2(list2):
        matches = list2[0]
        string = ""
        for i in range(0, len(list2[1:]), 2):
            doc_key = list2[1:][i]
            fields = list2[1:][i+1]
            string+= fields[1] + " "
        return string
    
    async def resp3(list3):
        list = []
        for i in range(0, len(list3[1:]), 2):
            doc_key = list3[1:][i]
            num = int(re.search(r"\d+", doc_key).group())
            fields = list3[1:][i+1][3]
            if fields in ['user', 'tool']:
                list.append({"role": fields, "content": list3[1:][i+1][5]})
                newindex = num+1
                try:
                    doc = json.loads(rb.execute_command("JSON.GET", f"chat:msg:{newindex}"))
                    text = doc["text"]
                    list.append({"role": 'assistant', "content": text})
                except Exception as e:
                    #if it goes thru user/tool but fails at this then it must be out of bounds
                    print(f"Exception: {e}")
                    list.append({"role": 'assistant', "content": " "})
            if fields == "assistant":
                list.append({"role": fields, "content": list3[1:][i+1][5]})
                newindex = num-1
                try:
                    doc = json.loads(rb.execute_command("JSON.GET", f"chat:msg:{newindex}"))
                    text = doc["text"]
                    list.append({"role": doc["role"], "content": text})
                except Exception as e:
                    #should never happen because no assistant talks in the 0 index
                    print(f"Exception: {e}")
        return list

    resp2_result = await resp2(list2)
    resp3_result = await resp3(list3)
    print(f"stm: {list2[0]}")
    print(f"chatlog: {list3[0]}")
    return resp2_result, resp3_result


    

async def classifier1(text, embedding):
    """currently no classifier 1 (long term memory)"""

async def classifier2(text, embedding):
    f"""classifier 2 (short term memory)
    schema for redis:
        Content: ,
        Source: ,
        VectorizeContent: ,

    """

    def ensure_index(dim=768):
        try:
            r.execute_command(
                "FT.CREATE", "stmem_idx",
                "ON", "JSON",
                "PREFIX", "1", "stmem:fact:",
                "SCHEMA",
                "$.embedding", "AS", "embedding", "VECTOR", "HNSW", "6",
                    "TYPE", "FLOAT32", "DIM", str(dim), "DISTANCE_METRIC", "COSINE",
                "$.text", "AS", "text", "TEXT"
            )
        except redis.ResponseError as e:
            if "Index already exists" not in str(e):
                raise

    def l2_normalize(v):
        v = v.astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    #add
    def upsert_fact(fid: str, text: str, emb: np.ndarray):
        key = f"stmem:fact:{fid}"
        doc = {
            "text": text,
            "embedding": l2_normalize(emb).astype(np.float32).tobytes()
        }
        r.execute_command("JSON.SET", key, "$", str(doc).replace("'", '"'))
        return key
    
    def knn_search(query_emb: np.ndarray, k=5):
        #top k
        q = l2_normalize(query_emb).astype(np.float32).tobytes()
        resp = r.execute_command(
            "FT.SEARCH", "stmem_idx",
            f"*=>[KNN {k} @embedding $vec]",
            "PARAMS", 2, "vec", q,
            "SORTBY", "__embedding_score",
            "DIALECT", 2,
            "RETURN", 1, "$.text"
        )
        return resp
    
    ensure_index()
    resp = knn_search(embedding)
    return resp

async def classifier3(text, embedding):
    """chatlogs"""
    def ensure_index(DIM=768):
        try:
            r.execute_command(
                "FT.CREATE", "chat_idx",
                "ON", "JSON",
                "PREFIX", "1", "chat:msg:",
                "SCHEMA",
                "$.ts", "AS", "ts", "TEXT",
                "$.role", "AS", "role", "TAG",
                "$.text", "AS", "text", "TEXT",
                "$.embedding", "AS", "embedding", "VECTOR", "HNSW", "6",
                    "TYPE", "FLOAT32",
                    "DIM", str(DIM),
                    "DISTANCE_METRIC", "COSINE"
            )
        except redis.ResponseError as e:
            if "Index already exists" not in str(e):
                raise

    def l2_normalize(v):
        v = v.astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)
    
    def knn_search(query_emb: np.ndarray, k=5):
        q = l2_normalize(query_emb).astype(np.float32).tobytes()
        resp = r.execute_command(
            "FT.SEARCH", "chat_idx",
            f"*=>[KNN {k} @embedding $vec]",
            "PARAMS", 2, "vec", q,
            "SORTBY", "__embedding_score",
            "DIALECT", 2,
            "LIMIT", 0, k,
            "RETURN", 4, "$", "$.ts", "$.role", "$.text"
        )
        return resp


    ensure_index()
    resp = knn_search(embedding)
    return resp
        



if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    async def main():
        await getRedis()

    asyncio.run(main())
