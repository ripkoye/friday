from ollama import Client
import os, json, time
import asyncio
import redis
import re, requests
from kokoro import KPipeline
import numpy as np
from datetime import datetime

from utils import jsoncleaner

REDIS_HOST= "127.0.0.1"
REDIS_PORT = 6379

#----Connection
r = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True, password='98679Ryna71@')
rb = redis.Redis(host="127.0.0.1", port=6379, decode_responses=False, password='98679Ryna71@')

systemPrompt = [
    {"role":"system","content": "Your name is Friday."
    "You are to only output in a json format."
    "No prose, no code fences, no markdown."
    f"This is the format of the json: Destination: Planner/Reasoner(string), Content: TLDR(string), Response: Talk(string) ."
    "You must decide if you need more information from the Planner or the Reasoner. If you understand the question well, you don't have to consult them. "
    "For example if you need certain information not available (like latest information on internet), "
    "you need to consult the planner since he has access to web crawling tools. "
    "Content component should be a TLDR of what the user input to help the planner. "
    "Response is the response you're giving to the user based on what they told you."
    "Keep the response conversation style like you're talking to somebody. "
    "You must always respond back to the reader."
    "This means to be short and concise."
    "Even facts should be short and concise, you don't have to go in much detail if the user is asking for a small fact."
    "If you don't need to consult the Planner or the Reasoner. Then you can put values 0 in Destination and Content."
    "Also right now the Reasoner has not been created yet. So don't do reasoner."
    "If you're getting a response back from the planner. You're only informing the user of what it got back, no need to send to anyone else."}
]

async def main():
    # Run both listeners concurrently
    await asyncio.gather(
        listen_for_triggers(),
        listen_for_planner_responses()
    )

async def listen_for_planner_responses():
    """Listen for planner responses and generate final responses"""
    pubsub = r.pubsub()
    pubsub.subscribe("planner_completed")
    
    print("[VoiceAgent] Listening for planner responses...")
    
    try:
        while True:
            message = pubsub.get_message(timeout=0.1)
            if message is None:
                await asyncio.sleep(0.1)
                continue
                
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    context_data = data['context']
                    planner_result = data['planner_result']
                    
                    print(f"[VoiceAgent] Got planner response for: {context_data['user_input']}")
                    
                    # Generate final response with planner results
                    planner_info = "\n".join([f"- {item.get('information')} from {item.get('url')}" for item in planner_result["CurrentInfo"]])
                    tasks = []
                    for item in planner_result["CurrentInfo"]:
                        info_text = f"{item.get('information')} from {item.get('url')}"
                        tasks.append(upsert_stmem(info_text))
                    
                    # Wait for all to complete
                    await asyncio.gather(*tasks)
                    enhanced_context = f"User asked: {context_data['user_input']}\n\nRelevant facts: {context_data['stm_facts']}\n\nPlanner found: {planner_info}"
                    
                    # Reconstruct messages for final response
                    messages = []
                    if not context_data['conversation_history'] or context_data['conversation_history'][0].get('role') != 'system':
                        messages.extend(systemPrompt)
                    
                    if context_data['conversation_history']:
                        messages.extend(context_data['conversation_history'])
                    
                    messages.append({"role": "user", "content": context_data['user_input']})
                    messages.append({"role": "assistant", "content": context_data['initial_response']})
                    messages.append({"role": "user", "content": f"The planner have searched the web and found information about: {context_data['user_input']}. The search results are included in the context. Please provide a final answer using this information. Do not call the planner again - just use the search results to give a complete response."})
                    
                    final_response = await get_final_response(messages, enhanced_context)
                
                    
                    # Send to TTS
                    r.publish("tts_requests", json.dumps({
                        "text": final_response,
                        "timestamp": time.time()
                    }))
                    
                    await upsert_chatlog("assistant", final_response)
                    
                except Exception as e:
                    print(f"[VoiceAgent] Error processing planner response: {e}")
                    import traceback
                    traceback.print_exc()
                    
    except KeyboardInterrupt:
        print("\n[VoiceAgent] Planner listener shutting down...")
        pubsub.close()

async def get_final_response(messages, enhanced_context):
    """Get final response from LLM with enhanced context"""
    try:
        # Add the enhanced context to the conversation
        enhanced_messages = messages + [{"role": "user", "content": f"Context: {enhanced_context}"}]
        
        response = requests.post(
            "http://localhost:8000/qwenCall",
            json={"history": enhanced_messages}
        )
        
        response_text = response.text
        print(f"[VoiceAgent] Final LLM response: {response_text}")
        
        # Parse the JSON response and extract just the Response field
        try:
            parsed_response = jsoncleaner.extract_json_block(response.text)[0]
            if isinstance(parsed_response, dict) and "Response" in parsed_response:
                final_text = parsed_response["Response"]
                print(f"[VoiceAgent] Extracted response: {final_text}")
                return final_text
            else:
                print(f"[VoiceAgent] No Response field found in: {parsed_response}")
                return response_text
        except Exception as parse_error:
            print(f"[VoiceAgent] Failed to parse JSON response: {parse_error}")
            return response_text
        
    except Exception as e:
        print(f"[VoiceAgent] Error getting final response: {e}")
        return "I encountered an error while processing the information."

async def listen_for_triggers():
    pubsub = r.pubsub()
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
                    if data.get('message'):
                        user = data.get('message')
                        string2 = data.get('resp2')
                        list3 = data.get('resp3')
                        
                        print(f"\n[VoiceAgent] FRIDAY TRIGGERED! Processing: '{user}'")
                        # Create async task for processing
                        asyncio.create_task(process_trigger(user, string2, list3))
                except Exception as e:
                    print()
                    #print(f"[VoiceAgent] Error processing trigger: {e}")
    except KeyboardInterrupt:
        #print("\n[VoiceAgent] Shutting down...")
        pubsub.close()



async def process_trigger(user, string2, list3):
    #calls api for qwen
    s = f"Here are some useful facts: {string2}. UserInput: {user}"

    # Ensure system prompt is at the beginning (only add if not already there)
    if not list3 or list3 is None:
        # If no conversation history, start fresh with system prompt
        messages = systemPrompt + [{"role": "user", "content": s}]
    elif list3[0].get("role") != "system":
        # If conversation history exists but no system prompt, add it
        messages = systemPrompt + list3 + [{"role": "user", "content": s}]
    else:
        # If system prompt already exists, just add user message
        messages = list3 + [{"role": "user", "content": s}]
    
    await upsert_chatlog("user", user)

    try:
        response = requests.post(
            "http://localhost:8000/qwenCall",
            json={"history": messages}
        )
        response_text = response.text
        print(f"[VoiceAgent] LLM Response: {response_text}")
        print(f"[VoiceAgent] Response type: {type(response_text)}")
        
        # Parse the JSON response
        try:
            llm_decision = jsoncleaner.extract_json_block(response_text)[0]
            print(f"[VoiceAgent] Parsed decision type: {type(llm_decision)}")
            print(f"[VoiceAgent] Parsed decision: {llm_decision}")
            
            destination = llm_decision.get("Destination", "0")
            content = llm_decision.get("Content", "0")
            response_msg = llm_decision.get("Response", "I'm not sure how to help with that.")
            r.publish("tts_requests", json.dumps({
                        "text": response_msg,
                        "timestamp": time.time()
                    }))
            await upsert_chatlog("assistant", response_msg)
            
            # Check if we need to call the planner
            if destination == "Planner":
                print(f"[VoiceAgent] Publishing planner request with: {content}")
                
                # Save the context for when planner returns
                context_data = {
                    "user_input": user,
                    "stm_facts": string2,
                    "conversation_history": list3,
                    "initial_response": response_msg,
                    "planner_content": content
                }
                
                # Publish planner request with context (non-blocking)
                import uuid
                request_id = str(uuid.uuid4())
                planner_request = {
                    "request_id": request_id,
                    "payload": content,
                    "context": context_data
                }
                
                r.publish("planner_requests_with_context", json.dumps(planner_request))
                print(f"[VoiceAgent] Published planner request {request_id}, will respond when complete")
                
                # Send immediate response to let user know we're working on it
            else:
                # No planner needed, send response directly
                print()
            
        except json.JSONDecodeError as e:
            print(f"[VoiceAgent] Failed to parse LLM response as JSON: {e}")
            # Fallback: send raw response to TTS
            r.publish("tts_requests", json.dumps({
                "text": response_text,
                "timestamp": time.time()
            }))
            
    except Exception as e:
        print(f"[VoiceAgent] Error: {e}")
        error_response = "Sorry, I encountered an error processing your request."
        r.publish("tts_requests", json.dumps({
            "text": error_response,
            "timestamp": time.time()
        }))
        await upsert_chatlog("assistant", error_response)


async def next_id(prefix: str) -> int:
    return r.incr(f"{prefix}:id")

async def upsert_stmem(text):
    doc_id = await next_id("stmem")
    response = requests.post(
        "http://localhost:8000/bert2embed",
        json={"texts": [text]},
    )
    result = response.json()
    embedding_vector = result["embeddings"][0]  # Extract first embedding
    
    # Convert to numpy array and normalize like TextClassifiers does
    embedding_array = np.array(embedding_vector, dtype=np.float32)
    normalized_embedding = embedding_array / (np.linalg.norm(embedding_array) + 1e-12)
    
    key = f"stmem:fact:{doc_id}"
    doc = { 
        "text": text, 
        "embedding": normalized_embedding.astype(np.float32).tolist()  # Convert to list instead of bytes
    }
    r.execute_command("JSON.SET", key, "$", json.dumps(doc))
    return key

async def upsert_chatlog(role, content):
    doc_id = await next_id("chatlog")
    key = f"chat:msg:{doc_id}"
    response = requests.post(
        "http://localhost:8000/bert2embed",
        json={"texts": [content]},
    )
    result = response.json()
    embedding_vector = result["embeddings"][0]
    
    # Convert to numpy array and normalize like TextClassifiers does
    embedding_array = np.array(embedding_vector, dtype=np.float32)
    normalized_embedding = embedding_array / (np.linalg.norm(embedding_array) + 1e-12)
    
    doc = { 
        "ts": datetime.now().isoformat(),
        "role": role, 
        "text": content, 
        "embedding": normalized_embedding.astype(np.float32).tolist()  # Convert to list instead of bytes
    }
    r.execute_command("JSON.SET", key, "$", json.dumps(doc))
    return key


if __name__ == '__main__':
    asyncio.run(main())        
