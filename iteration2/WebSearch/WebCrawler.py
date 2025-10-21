import scrapy
from scrapy.crawler import CrawlerRunner
from scrapy import signals
from WebSearch.Seeds import Seeds
import asyncio
from bs4 import BeautifulSoup
import re

class MiniSpider(scrapy.Spider):
    """
    A Scrapy spider that extracts text from web pages based on keywords.
    """
    name = "mini"
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "AUTOTHROTTLE_ENABLED": True,
        "DOWNLOAD_DELAY": 0.1,        # Faster: 0.1s instead of 0.5s
        "DEPTH_LIMIT": 1,             # Shallower: 1 level instead of 2
        "LOG_LEVEL": "ERROR",
        "CLOSESPIDER_PAGECOUNT": 20,  # Fewer pages: 20 instead of 100
        "DOWNLOAD_TIMEOUT": 10,       # Timeout slow requests after 10s
    }

    def __init__(self, start_urls: list[str], keywords: list[str] = None, text_collector: list = None, **kw):
        """
        Initializes the MiniSpider.

        Args:
            start_urls (list[str]): List of URLs to start crawling.
            keywords (list[str]): List of keywords to filter results.
            text_collector (list): A shared list to collect extracted data.

        Returns:
            None
        """
        super().__init__(**kw)
        self.start_urls = start_urls
        self.keywords = [k.lower() for k in (keywords or [])]
        self.text_collector = text_collector  # external list reference

    def extract_text(self, response) -> str:
        """
        Extracts clean text content from the response using BeautifulSoup.

        Args:
            response (scrapy.http.Response): The response object.

        Returns:
            str: The extracted and cleaned text.
        """
        try:
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, and other unwanted tags
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                tag.decompose()
            
            # Try to get main content first
            main_content = soup.find(['main', 'article']) or soup.find('body') or soup
            
            # Extract text and clean it up
            text = main_content.get_text(separator=' ', strip=True)
            
            # Remove extra whitespace and clean up
            text = re.sub(r'\s+', ' ', text)  # Multiple spaces -> single space
            text = re.sub(r'\n+', ' ', text)  # Multiple newlines -> space
            
            # Limit length to avoid overwhelming the LLM
            return text[:1000] if text else ""
            
        except Exception as e:
            # Fallback to original XPath method if BeautifulSoup fails
            txt = response.xpath("normalize-space(string(//main|//article|//body))").get() or ""
            return " ".join(txt.split())[:1000]

    def parse(self, response):
        """
        Parses the response and collects relevant data.

        Args:
            response (scrapy.http.Response): The response object.

        Returns:
            None
        """
        body_text = self.extract_text(response).lower()
        is_hit = True if not self.keywords else any(k in body_text for k in self.keywords)

        if is_hit:
            # Append a dictionary with the URL and extracted text to the shared collector
            self.text_collector.append({
                "url": response.url,
                "information": body_text
            })

        # Follow valid links only
        for href in response.css("a::attr(href)").getall():
            if href.startswith("http://") or href.startswith("https://"):
                yield response.follow(href, callback=self.parse)

class WebCrawler:
    def __init__(self, queries: list[str], keywords: list[str] = None, url_count: int = 5):
        """
        Initializes the WebCrawler.

        Args:
            queries (list[str]): List of queries to search.
            keywords (list[str]): List of keywords to filter results.
            url_count (int): Number of URLs to gather per query.
        """
        print(f"DEBUG: WebCrawler init with queries: {queries}, url_count: {url_count}")
        seed = Seeds(queries, url_count)
        self.urls = seed.urls
        self.keywords = keywords or []
        print(f"DEBUG: Seeds generated {len(self.urls)} URLs")
        if len(self.urls) == 0:
            print("ERROR: Seeds returned no URLs! Check SearxNG connection.")
        elif len(self.urls) < url_count:
            print(f"WARNING: Only got {len(self.urls)} URLs, expected {url_count}")

    def run(self) -> list[dict]:
        """
        Runs the spider in a subprocess to avoid reactor conflicts.

        Returns:
            list[dict]: A list of dictionaries containing URLs and extracted information.
        """
        print(f"WebCrawler starting with {len(self.urls)} URLs and keywords: {self.keywords}")
        
        # Use subprocess to completely isolate each crawler
        import subprocess
        import tempfile
        import json
        
        # Create temporary files for input/output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
            input_data = {
                'urls': self.urls,
                'keywords': self.keywords
            }
            json.dump(input_data, input_file)
            input_path = input_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
            output_path = output_file.name
        
        # Create subprocess script
        script_content = f'''
import sys
import json
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup
import re
import scrapy

class SubprocessSpider(scrapy.Spider):
    name = "subprocess"
    custom_settings = {{
        "ROBOTSTXT_OBEY": True,
        "AUTOTHROTTLE_ENABLED": True,
        "DOWNLOAD_DELAY": 0.1,
        "DEPTH_LIMIT": 1,
        "LOG_LEVEL": "ERROR",
        "CLOSESPIDER_PAGECOUNT": 20,
        "DOWNLOAD_TIMEOUT": 10,
    }}

    def __init__(self, start_urls, keywords, text_collector, **kw):
        super().__init__(**kw)
        self.start_urls = start_urls
        self.keywords = [k.lower() for k in (keywords or [])]
        self.text_collector = text_collector

    def extract_text(self, response):
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                tag.decompose()
            main_content = soup.find(['main', 'article']) or soup.find('body') or soup
            text = main_content.get_text(separator=' ', strip=True)
            text = re.sub(r'\\s+', ' ', text)
            text = re.sub(r'\\n+', ' ', text)
            return text[:1000] if text else ""
        except:
            return ""

    def parse(self, response):
        body_text = self.extract_text(response).lower()
        is_hit = True if not self.keywords else any(k in body_text for k in self.keywords)
        if is_hit:
            self.text_collector.append({{
                "url": response.url,
                "information": body_text
            }})

# Load input
with open("{input_path}", 'r') as f:
    data = json.load(f)

text_collector = []
process = CrawlerProcess()
process.crawl(SubprocessSpider, start_urls=data['urls'], keywords=data['keywords'], text_collector=text_collector)
process.start()

# Save results
with open("{output_path}", 'w') as f:
    json.dump(text_collector, f)
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
            script_file.write(script_content)
            script_path = script_file.name
        
        try:
            # Run subprocess
            result = subprocess.run([
                'python', script_path
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Load results
                with open(output_path, 'r') as f:
                    text_collector = json.load(f)
            else:
                print(f"Subprocess error: {result.stderr}")
                text_collector = []
                
        except subprocess.TimeoutExpired:
            print("Crawler timed out after 60 seconds")
            text_collector = []
        except Exception as e:
            print(f"Subprocess error: {e}")
            text_collector = []
        finally:
            # Cleanup temp files
            import os
            try:
                os.unlink(input_path)
                os.unlink(output_path) 
                os.unlink(script_path)
            except:
                pass
        
        print(f"WebCrawler finished! Collected {len(text_collector)} items")
        if len(text_collector) == 0:
            print("DEBUG: No items collected - check subprocess output above")
        else:
            print(f"DEBUG: Sample collected item: {text_collector[0]}")
        
        return text_collector