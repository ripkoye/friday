from ollama import Client
import os, json
import asyncio, aiohttp
from search.WebCrawler import WebCrawler
from concurrent.futures import ThreadPoolExecutor
import redis

import re

# Redis-based communication - no more Unix sockets needed


async def handle_redis_requests():
    """
    Listen for planner requests via Redis pub/sub and handle them
    """
    r = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True, password='98679Ryna71@')
    pubsub = r.pubsub()
    pubsub.subscribe("planner_requests")
    
    print("[PlannerAgent] Listening for Redis requests...")
    
    try:
        for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    request_data = json.loads(message['data'])
                    request_id = request_data['request_id']
                    payload = request_data['payload']
                    
                    print(f"[PlannerAgent] Processing request {request_id}")
                    
                    # Instantiate PlannerAgent and run its workflow
                    planner_agent = PlannerAgent(payload)
                    result = await planner_agent.workflow()
                    
                    # Send response back via Redis
                    r.publish(f"planner_response_{request_id}", json.dumps(result))
                    
                except Exception as e:
                    print(f"[PlannerAgent] Error processing request: {e}")
                    
    except KeyboardInterrupt:
        print("\n[PlannerAgent] Shutting down...")
        pubsub.close()

async def main():
    """
    Starts the Redis listener for PlannerAgent requests.
    """
    await handle_redis_requests()


class PlannerAgent:
    """
    The PlannerAgent is responsible for coordinating WebAgents to gather information
    and iteratively refine the corpus until sufficient information is collected.
    """
    systemPrompt = (
        "You are a planner agent. CRITICAL: You MUST output ONLY valid JSON, nothing else. "
        "The workflow is: text goes into the Voice Agent and gets sent to you. "
        "Your job is to gather/call necessary tools to aid the Reasoner agent or Voice Agent. "
        "\n\nYour JSON output MUST ALWAYS have this exact structure:\n"
        "{\n"
        "  \"ContentIsSufficient\": true/false,\n"
        "  \"WebAgentCount\": number,\n"
        "  \"SearchKeywords\": [[\"keyword1\", \"keyword2\"], [\"keyword3\", \"keyword4\"]],\n"
        "  \"Queries\": [[\"query1\", \"query2\"], [\"query3\", \"query4\"]],\n"
        "  \"CurrentInfo\": [{\"info\": \"text\", \"url\": \"url\"}, {\"info\": \"text\", \"url\": \"url\"}],\n"
        "  \"Destination\": \"VoiceAgent\" or \"Reasoner\"\n"
        "}\n\n"
        "Rules:\n"
        "- ContentIsSufficient: true if you have enough info in the CurrentInfo entry, false if you need more\n"
        "- WebAgentCount: how many web crawlers to run (1-3 recommended)\n"
        "- SearchKeywords: keywords for each web agent (as nested arrays)\n"
        "- Queries: search queries for each web agent (as nested arrays)\n"
        "- CurrentInfo: only add NEW information relevant to user's question\n"
        "- Destination: \"VoiceAgent\" for simple questions, \"Reasoner\" for complex analysis\n"
        "\nIMPORTANT: Output ONLY the JSON object, no explanations or extra text."
    )
    messages = []
    contentIsSuff = True
    currentInfo = []
    

    def __init__(self, VoiceAgentMessage: dict):
        """
        Initializes the PlannerAgent.

        Args:
            VoiceAgentMessage (dict): The initial message from the VoiceAgent.

        Returns:
            None
        """
        self.client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        self.model = "qwen2.5"

        self.messages = [
            {"role": "system", "content": self.systemPrompt},
        ]
        print(f"DEBUG: Initialized with system prompt: {self.systemPrompt[:200]}...")
        print(f"DEBUG: Initial messages array: {len(self.messages)} messages")
        self.resp = self.call(f"The Voice Agent said this: {json.dumps(VoiceAgentMessage)}")

        

    def call(self, input: str) -> str:
        """
        Sends a message to the client and retrieves the response.

        Args:
            input (str): The input message to send.

        Returns:
            str: The response message from the client.
        """
        print(f"DEBUG: About to call LLM with {len(self.messages)} messages")
        print(f"DEBUG: System prompt still there? {self.messages[0]['role'] == 'system'}")
        print(f"DEBUG: System prompt content: {self.messages[0]['content'][:100]}...")
        
        self.messages.append({"role": "user", "content": input})
        print(f"DEBUG: Added user message. Now have {len(self.messages)} messages")
        
        # Add JSON instruction to user message for extra emphasis
        enhanced_messages = self.messages.copy()
        enhanced_messages[-1]["content"] += "\n\nRemember: Respond with ONLY valid JSON, no other text."
        
        resp = self.client.chat(
            model=self.model,
            messages=enhanced_messages,
            stream=False,
            keep_alive="5m",
            options={"temperature": 0.3,  "stop": ["<|im_end|>", "<|im_start|>"]}
        )
        answer = resp["message"]["content"]
        print(f"DEBUG: Raw LLM response: '{answer}'")
        self.messages.append({"role": "assistant", "content": answer})
        print(f"DEBUG: Added assistant response. Now have {len(self.messages)} messages")
        return answer

    async def workflow(self):
        """
        Iteratively gathers information using WebAgents until the corpus is sufficient.

        Returns:
            list[dict]: The final `CurrentInfo` containing the gathered information.
        """
        outputDict = self.extract_json_block(self.resp)  # Expected to be a dict
        self.contentIsSuff = outputDict["ContentIsSufficient"]  # bool
        info = []

        print(self.contentIsSuff)
        while not self.contentIsSuff:
            web_agent_count = outputDict["WebAgentCount"]  # int
            print(f"Starting {web_agent_count} web crawlers...")
            print("Queries", outputDict["Queries"])
            print("SearchKeywords", outputDict["SearchKeywords"])
            
            # Run WebAgents concurrently
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                tasks = [
                    loop.run_in_executor(
                        executor,
                        self.webAgent,
                        outputDict["Queries"][i],
                        outputDict["SearchKeywords"][i]
                    )
                    for i in range(web_agent_count)
                ]
            print("Waiting for crawlers to complete...")
            results = await asyncio.gather(*tasks)
            print(f"Crawlers finished! Got {len(results)} result sets")

            # Flatten and format results
            flattened_results = [
                f"URL: {item['url']}, Information: {item['information']}"
                for sublist in results for item in sublist
            ]

            newprompt = "Corpus collected from WebAgents: " + "\n".join(flattened_results)

            output = self.call(newprompt)
            print(f"LLM Output: {output}")  # Show first 200 chars
            try:
                jsonOutput = self.extract_json_block(output)
            except ValueError as e:
                print(f"JSON parsing failed: {e}")
                print(f"Full LLM output: {output}")
                # Default to stopping the loop if JSON parsing fails
                self.contentIsSuff = True
                info = []
                break
            self.contentIsSuff = jsonOutput["ContentIsSufficient"]
            info = jsonOutput["CurrentInfo"]
            print("info is", info)

        return info
            
        

    def webAgent(self, queries: list[str], keywords: list[str]) -> list[dict]:
        """
        Executes a WebCrawler to gather information.

        Args:
            queries (list[str]): List of queries to search.
            keywords (list[str]): List of keywords to filter results.

        Returns:
            list[dict]: A list of dictionaries containing URLs and extracted information.
        """
        crawl = WebCrawler(queries, keywords)
        information = crawl.run()  # Synchronous call
        return information
    
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
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())