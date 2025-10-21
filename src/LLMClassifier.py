from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import redis, time, json

r = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True, password='98679Ryna71@')

STREAM = "transcript:session1"
last_id = "0"  # Start from beginning to read existing messages

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LLMClassifier] Starting on device: {DEVICE}")

    print("[LLMClassifier] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained("../models/friday_agent_model")
    print("[LLMClassifier] Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "../models/friday_agent_model"
    ).to(DEVICE)
    print("[LLMClassifier] Model loaded successfully!")

    string = ""

    while True:
        resp = r.xread({STREAM: last_id,}, block = 5000, count=10)
        if not resp:
            continue

        for stream, messages in resp:
            for msg_id, fields in messages:
                text = fields['text']
                start = fields['start']
                end = fields['end']
                string+=text
                last_id = msg_id  # Update to continue from this message
                print(stream, msg_id, fields)

        #text = input("type something: ")
        # Truncate text to fit model's max sequence length (512 tokens)
        inputs = tok(string, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}  # move to GPU/CPU

        with torch.no_grad():
            outputs = model(**inputs)

        pred = outputs.logits.argmax(-1).item()
        print("Predicted label:", pred)
        
        # Temporary fix: manually check for Friday triggers
        if "friday" in string.lower() or pred == 1:
            print("[LLMClassifier] TRIGGER DETECTED! Sending to VoiceAgent...")
            # Send the accumulated transcript to VoiceAgent via Redis
            r.publish("voice_agent_trigger", json.dumps({
                "trigger": True,
                "transcript": string,
                "timestamp": time.time()
            }))
            r.delete("transcript:session1")
            string = ""
            last_id = "0"  # Reset to beginning