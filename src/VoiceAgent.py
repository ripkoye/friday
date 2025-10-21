from ollama import Client
import os, json, time
import asyncio
import redis
import re
from kokoro import KPipeline
import numpy as np



# Redis-based communication - no more Unix sockets needed

class VoiceAgent:
    messages = [
        {"role":"system","content": "Your name is Friday."
        "You are to only output in a json format. "
        "This is the format: {Destination: Planner/Reasoner, Content: TLDR, Response: Talk}. "
        "You must decide if the piece of text should be sent you a planner or a reasoner or none. "
        "For example if you need certain information not available (like latest information on internet), "
        "you need to consult the planner since he has access to web crawling tools. "
        "Let's say you need a complex reasoning task like math related, you need to consult the reasoner. "
        "Content component should be a TLDR of what the user input to help the planner. "
        "Response is the response you're giving to the user based on what they told you. "
        "Keep the response conversation style like you're talking to somebody. This means to be short and concise."
        "Even facts should be short and concise, you don't have to go in much detail if the user is asking for a small fact."
        "If you don't need to consult the Planner/Reasoner. Then you can put values 0 in Destination and Content."
        "Also right now the Reasoner has not been created yet. So don't do reasoner."}
    ]

    def __init__(self):
        self.client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        self.MODEL = "qwen2.5"
        self.redis_client = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True, password='98679Ryna71@')
        
    async def start(self):
        """Start the VoiceAgent listener"""
        await self.listen_for_triggers()


    async def listen_for_triggers(self):
        """Listen for trigger messages from LLMClassifier via Redis pub/sub"""
        print("[VoiceAgent] Waiting for Friday triggers...")
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe("voice_agent_trigger")
        
        try:
            while True:
                # Use get_message with timeout to make it non-blocking
                message = pubsub.get_message(timeout=0.1)
                if message is None:
                    await asyncio.sleep(0.1)  # Yield control
                    continue
                    
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        if data.get('trigger'):
                            transcript = data.get('transcript', '')
                            print(f"\n[VoiceAgent] FRIDAY TRIGGERED! Processing: '{transcript}'")
                            # Create async task for processing
                            asyncio.create_task(self.process_trigger(transcript))
                    except Exception as e:
                        print()
                        #print(f"[VoiceAgent] Error processing trigger: {e}")
        except KeyboardInterrupt:
            #print("\n[VoiceAgent] Shutting down...")
            pubsub.close()

    async def process_trigger(self, transcript):
        """Process a Friday trigger asynchronously"""
        try:
            # Run the blocking call method in a thread executor
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.call, transcript)
        except Exception as e:
            print(f"[VoiceAgent] Error processing trigger: {e}")

    def call(self, userInput):
        self.messages.append({"role":"user", "content": userInput})
        stream = self.client.chat(model=self.MODEL, messages=self.messages, stream=True, keep_alive="5m", options={"temperature":0.3, "stop": ["<|im_end|>", "<|im_start|>"]})
        corpus = ""
        for chunk in stream:
            corpus+= chunk["message"]["content"]
            print(chunk["message"]["content"], end="", flush=True)
        print()
        self.messages.append({"role":"assistant", "content": corpus})
        try:
            output = self.extract_json_block(corpus)
            self.redis_client.publish("tts_requests", json.dumps({
                "text": output['Response'],
                "timestamp": time.time()
            }))
        except Exception as e:
            print(f"could not parse json: {e}")
            return
        
        if(output["Destination"]=="Planner"):
            # Handle planner request asynchronously
            asyncio.create_task(self.handle_planner_request(output))

    async def handle_planner_request(self, output):
        """Handle planner request asynchronously so VoiceAgent can continue processing new triggers"""
        try:
            print("Starting planner request (async)...")
            
            # Call planner (this will block this task but not the main listener)
            reply = self.planner(output)
            
            print("Planner response received, generating user response...")
            
            # Feed planner data back to LLM for a proper user response
            planner_prompt = f"Based on this information I gathered: {reply}, please provide a helpful yet concise response to the user's question."
            self.messages.append({"role": "user", "content": planner_prompt})
            
            # Get LLM's final response to user
            resp = self.client.chat(
                model=self.MODEL, 
                messages=self.messages, 
                stream=False, 
                keep_alive="5m", 
                options={"temperature": 0.3, "stop": ["<|im_end|>", "<|im_start|>"]}
            )
            
            final_answer = resp["message"]["content"]
            self.messages.append({"role": "assistant", "content": final_answer})
            
            print("\nFinal planner response:")
            print(final_answer)
            
            # Send planner response to TTS
            self.redis_client.publish("tts_requests", json.dumps({
                "text": final_answer,
                "timestamp": time.time()
            }))
            
        except Exception as e:
            print(f"Error in planner request: {e}")

    def extract_json_block(self, s: str) -> dict:
        s = s.strip()
        # strip code fences if present
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
        # try direct
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # fallback: first {...}
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            raise ValueError("No JSON object found in model output")
        return json.loads(m.group(0))

            

    def planner(self, json_payload=None):
        """
        Send JSON payload to Planner via Redis and wait for reply
        """
        import uuid
        
        # Use provided payload or try to extract from messages
        if json_payload:
            payload = json_payload
        else:
            try:
                payload_text = self.messages[-1]["content"]           # JSON string from your LLM
                payload = json.loads(payload_text)                     # make sure it's valid JSON
            except Exception as e:
                print(f"\n[planner] Could not parse last assistant message as JSON: {e}")
                return None

        # Create unique request ID for response tracking
        request_id = str(uuid.uuid4())
        request_data = {
            "request_id": request_id,
            "payload": payload,
            "timestamp": time.time()
        }
        
        # Send request to planner via Redis
        self.redis_client.publish("planner_requests", json.dumps(request_data))
        
        # Wait for response on dedicated channel
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(f"planner_response_{request_id}")
        
        try:
            # Wait for response with timeout
            for message in pubsub.listen():
                if message['type'] == 'message':
                    reply = json.loads(message['data'])
                    print(f"\n[planner reply] {reply}")
                    pubsub.close()
                    return reply
        except Exception as e:
            print(f"\n[planner] Error waiting for reply: {e}")
            pubsub.close()
            return None
        

    def clearMessages(self):
        self.messages = []
    


if __name__ == "__main__":
    async def main():
        agent = VoiceAgent()
        await agent.start()
    
    asyncio.run(main())