import redis
import json
import asyncio
import requests
import re
from WebSearch.WebCrawler import WebCrawler
from concurrent.futures import ThreadPoolExecutor
from utils import jsoncleaner


# Redis connection
r = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True, password='98679Ryna71@')

# System prompt for planner
PLANNER_PROMPT = """You are a planner agent. You MUST output ONLY valid JSON.

When the Voice Agent asks for information, decide what to search for and how many URLs to gather.

JSON format:
{
  "ContentIsSufficient": True/False,
  "WebAgentCount": 1,
  "SearchKeywords": [["keyword1", "keyword2"]],
  "Queries": [["query1", "query2"]],
  "UrlCount": n,
  "CurrentInfo": [{"info": "text", "url": "url"}],
  "Destination": "VoiceAgent"
}

Rules:
- ContentIsSufficient: true if you have enough info, false if you need to search
- WebAgentCount: how many web crawlers to run (1-2 max for minimal version)
- SearchKeywords: keywords for filtering results
- Queries: search queries for each crawler
- UrlCount: how many URLs to gather per query (3-15 based on task complexity)
  * Simple facts: 3-5 URLs
  * General topics: 5-8 URLs  
  * Complex research: 20-40 URLs
- CurrentInfo: relevant information found
- Destination: always "VoiceAgent" for now

Output ONLY the JSON object, no explanations."""




async def handle_planner_requests():
    """Listen for planner requests via Redis pub/sub"""
    pubsub = r.pubsub()
    pubsub.subscribe("planner_requests_with_context")
    
    print("[Planner] Listening for Redis requests with context...")
    
    try:
        while True:
            # Use get_message with timeout to make it non-blocking
            message = pubsub.get_message(timeout=0.1)
            if message is None:
                await asyncio.sleep(0.1)  # Yield control
                continue
                
            if message['type'] == 'message':
                try:
                    request_data = json.loads(message['data'])
                    request_id = request_data['request_id']
                    payload = request_data['payload']
                    context = request_data['context']
                    
                    print(f"[Planner] Processing request {request_id}")
                    
                    # Process the request asynchronously
                    asyncio.create_task(process_single_request_with_context(request_id, payload, context))
                    
                except Exception as e:
                    print(f"[Planner] Error processing request: {e}")
                    
    except KeyboardInterrupt:
        print("\n[Planner] Shutting down...")
        pubsub.close()

async def process_single_request_with_context(request_id, payload, context):
    """Process a single planner request with context and send response to VoiceAgent"""
    try:
        # Process the request
        result = await process_planner_request(payload)
        
        print(f"result: {result}")

        # Send response back to VoiceAgent with context
        response_data = {
            "context": context,
            "planner_result": result
        }
        r.publish("planner_completed", json.dumps(response_data))
        print(f"[Planner] Sent response to VoiceAgent for request {request_id}")
        
    except Exception as e:
        print(f"[Planner] Error processing request {request_id}: {e}")
        # Send error response
        error_result = {"ContentIsSufficient": True, "CurrentInfo": [], "Destination": "VoiceAgent", "error": str(e)}
        response_data = {
            "context": context,
            "planner_result": error_result
        }
        r.publish("planner_completed", json.dumps(response_data))

async def process_planner_request(payload):
    """Process a planner request and return results"""
    try:
        # Get LLM decision on what to search for
        llm_response = await call_llm(payload)
        print(f"[Planner] Raw LLM response: {repr(llm_response)}")
        decision = jsoncleaner.extract_json_block(llm_response)[0]
        print(f"[Planner] Parsed decision: {decision}")
        
        # If we need to search, run web crawlers
        if not decision.get("ContentIsSufficient", True):
            web_results = await run_web_crawlers(decision)
            decision["CurrentInfo"] = web_results
            decision["ContentIsSufficient"] = True
        
        return decision
        
    except Exception as e:
        print(f"[Planner] Error in process_planner_request: {e}")
        return {"ContentIsSufficient": True, "CurrentInfo": [], "Destination": "VoiceAgent"}

async def call_llm(payload):
    """Call the LLM API for planner decisions"""
    try:
        messages = [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": f"Voice Agent request: {json.dumps(payload)}"}
        ]
        
        response = requests.post(
            "http://localhost:8000/qwenCall",
            json={"history": messages}
        )
        return response.text
        
    except Exception as e:
        print(f"[Planner] LLM call error: {e}")
        return '{"ContentIsSufficient": true, "CurrentInfo": [], "Destination": "VoiceAgent"}'



async def run_web_crawlers(decision):
    """Run web crawlers based on LLM decision"""
    try:
        web_count = decision.get("WebAgentCount", 1)
        queries = decision.get("Queries", [[]])
        keywords = decision.get("SearchKeywords", [[]])
        url_count = decision.get("UrlCount", 5)
        
        # Limit to 2 crawlers max for minimal version
        web_count = min(web_count, 2)
        
        # Limit URL count to reasonable range
        url_count = max(3, min(url_count, 15))
        
        print(f"[Planner] Running {web_count} web crawlers with {url_count} URLs each...")
        
        # Run crawlers concurrently
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            tasks = [
                loop.run_in_executor(
                    executor,
                    run_single_crawler,
                    queries[i] if i < len(queries) else [],
                    keywords[i] if i < len(keywords) else [],
                    url_count
                )
                for i in range(web_count)
            ]
        
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        flattened = []
        for sublist in results:
            flattened.extend(sublist)
        
        print(f"[Planner] Collected {len(flattened)} items from web crawlers")
        return flattened
        
    except Exception as e:
        print(f"[Planner] Web crawler error: {e}")
        return []

def run_single_crawler(queries, keywords, url_count=5):
    """Run a single web crawler (synchronous)"""
    try:
        crawler = WebCrawler(queries, keywords, url_count)
        return crawler.run()
    except Exception as e:
        print(f"[Planner] Single crawler error: {e}")
        return []

async def main():
    """Main entry point"""
    await handle_planner_requests()

if __name__ == "__main__":
    asyncio.run(main())
