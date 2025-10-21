#!/usr/bin/env python3
import requests
import time
import random

HOST = "http://127.0.0.1:8888"  # Change if your SearxNG port differs
TIMEOUT = 15
MAX_PAGES = 5
MAX_RETRIES = 5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{HOST}/",
}


class Seeds:
    """
    A class to collect URLs from SearxNG for a given list of queries.
    """

    def __init__(self, queries: list[str], url_count: int = 5):
        """
        Initializes the Seeds class and collects URLs for the given queries.

        Args:
            queries (list[str]): A list of search queries.
            url_count (int): Number of URLs to gather per query.

        Returns:
            None
        """
        self.urls = self.collect_urls(queries, url_count)

    def collect_urls(self, queries: list[str], url_count: int = 5) -> list[str]:
        """
        Collects URLs from SearxNG for the given queries.

        Args:
            queries (list[str]): A list of search queries.

        Returns:
            list[str]: A sorted list of unique URLs collected from SearxNG.
        """
        urls = set()  # Use a set to ensure unique URLs
        session = requests.Session()

        # Warm-up request to the SearxNG host
        try:
            session.get(f"{HOST}/", timeout=5)
        except Exception as e:
            print(f"[warn] Failed to connect to {HOST}: {e}")

        # Iterate over each query
        for query in queries:
            for page in range(1, MAX_PAGES + 1):
                attempt = 0
                while attempt < MAX_RETRIES:
                    try:
                        # Make a GET request to the SearxNG search endpoint
                        response = session.get(
                            f"{HOST}/search",
                            params={"q": query, "format": "json", "pageno": page},
                            headers=HEADERS,
                            timeout=TIMEOUT,
                        )

                        # Handle rate-limiting (HTTP 429)
                        if response.status_code == 429:
                            retry_after = int(response.headers.get("Retry-After", "0") or 0)
                            wait = retry_after if retry_after > 0 else (2 ** attempt) + random.uniform(0, 0.5)
                            print(f"[429] Backing off {wait:.2f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                            time.sleep(wait)
                            attempt += 1
                            continue

                        # Raise an exception for other HTTP errors
                        response.raise_for_status()

                        # Parse the JSON response
                        data = response.json()
                        results = data.get("results", [])

                        # Extract URLs from the results
                        for item in results:
                            url = item.get("url")
                            if url:
                                urls.add(url)

                        # Pacing between requests
                        time.sleep(random.uniform(0.6, 1.2))

                        # Stop pagination if no results are returned
                        if not results:
                            break

                        # Break out of the retry loop on success
                        break

                    except Exception as e:
                        print(f"[warn] Query={query!r}, Page={page}, Attempt={attempt + 1}: {e}")
                        time.sleep(0.5)
                        attempt += 1

        # Return a sorted list of unique URLs, limited by url_count
        sorted_urls = sorted(urls)
        limited_urls = sorted_urls[:url_count]
        print(f"[Seeds] Collected {len(sorted_urls)} URLs, returning {len(limited_urls)} (limit: {url_count})")
        return limited_urls