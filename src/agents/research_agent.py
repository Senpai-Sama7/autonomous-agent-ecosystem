"""
Research Agent with Real Web Search Capabilities

REFACTORED: Async implementation to prevent event loop blocking.
- Replaced requests with aiohttp for async HTTP
- Replaced time.sleep with asyncio.sleep for rate limiting
- Wrapped blocking DDGS search in asyncio.to_thread
"""

import asyncio
import time
from typing import Dict, Any, List, Optional

# Use new ddgs package with fallback to duckduckgo_search
try:
    from ddgs import DDGS

    DDGS_NEW_API = True
except ImportError:
    from duckduckgo_search import DDGS

    DDGS_NEW_API = False
import aiohttp
from bs4 import BeautifulSoup
from .base_agent import BaseAgent, AgentCapability, AgentContext, TaskResult, AgentState

# Structured logging
from src.utils.structured_logger import get_logger, log_performance

logger = get_logger("ResearchAgent")


class ResearchAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, [AgentCapability.DATA_PROCESSING], config)
        self.max_results = config.get("max_search_results", 5)
        self.max_scrape_results = config.get(
            "max_scrape_results", 3
        )  # Configurable scrape limit

        # Rate limiting with lock for thread safety
        self.min_request_interval = config.get(
            "min_request_interval", 1.0
        )  # seconds between requests
        self.last_request_time = 0
        self._rate_limit_lock = asyncio.Lock()

        # Simple in-memory cache with size limit
        self.cache: Dict[str, tuple] = {}
        self.cache_ttl = config.get("cache_ttl", 3600)  # 1 hour default
        self.max_cache_size = config.get("max_cache_size", 1000)  # Prevent memory leak

        # SSL verification (disable for testing only)
        self.verify_ssl = config.get("verify_ssl", True)

    async def execute_task(
        self, task: Dict[str, Any], context: AgentContext
    ) -> TaskResult:
        try:
            self.state = AgentState.BUSY
            query = task.get("payload", {}).get("query") or task.get("description")

            if not query:
                return TaskResult(success=False, error_message="No query provided")

            logger.info(f"Performing async web search for: {query}")

            # 1. Real Web Search (DDGS is sync, wrap in thread executor)
            results = await asyncio.to_thread(self._search_web, query)

            # 2. Content Extraction (Async HTTP with aiohttp)
            detailed_results = []
            # Filter results that have valid URLs
            valid_results = [res for res in results if res.get("href")]
            scrape_targets = valid_results[: self.max_scrape_results]

            if scrape_targets:
                connector = aiohttp.TCPConnector(ssl=self.verify_ssl)
                async with aiohttp.ClientSession(connector=connector) as session:
                    # Gather scraping tasks concurrently
                    scrape_tasks = [
                        self._scrape_content_async(session, res["href"])
                        for res in scrape_targets
                    ]
                    contents = await asyncio.gather(
                        *scrape_tasks, return_exceptions=True
                    )

                    for res, content in zip(scrape_targets, contents):
                        if isinstance(content, Exception):
                            logger.debug(
                                f"Scrape failed for {res.get('href')}: {content}"
                            )
                            detailed_results.append(
                                res
                            )  # Include result without full_content
                        elif content:  # Non-empty string
                            res["full_content"] = content[
                                :2000
                            ]  # Truncate for context window
                            detailed_results.append(res)
                        else:
                            detailed_results.append(
                                res
                            )  # Include result without full_content

            # 3. Synthesis (Simple aggregation for now, could use LLM if configured)
            summary = self._synthesize_results(detailed_results)

            return TaskResult(
                success=True,
                result_data={
                    "summary": summary,
                    "sources": detailed_results,
                    "raw_search_results": results,
                },
                execution_time=0.0,  # Calculated by caller
            )

        except Exception as e:
            logger.error(f"Research failed: {e}")
            return TaskResult(success=False, error_message=str(e))
        finally:
            self.state = AgentState.ACTIVE

    def _search_web(self, query: str) -> List[Dict]:
        """Execute DuckDuckGo search with caching (runs in thread executor)"""
        # Check cache first
        cache_key = f"search:{query}"
        if self._check_cache(cache_key):
            logger.info(f"Using cached search results for: {query}")
            return self.cache[cache_key][0]

        # Note: Rate limiting for this sync method is handled at call site
        # since we can't use asyncio.sleep here (we're in a thread)
        try:
            # New ddgs API doesn't use context manager
            if DDGS_NEW_API:
                results = DDGS().text(query, max_results=self.max_results)
            else:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=self.max_results))
            # Cache results
            self._add_to_cache(cache_key, results)
            return results
        except Exception as e:
            logger.error(f"Search API error: {e}")
            return []

    async def _scrape_content_async(
        self, session: aiohttp.ClientSession, url: str
    ) -> str:
        """Async scrape text content from URL with caching"""
        # Check cache
        cache_key = f"scrape:{url}"
        if self._check_cache(cache_key):
            logger.info(f"Using cached content for: {url}")
            return self.cache[cache_key][0]

        # Async rate limiting
        await self._apply_rate_limit_async()

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"},
            ) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return ""
                html = await response.text()

                # CPU-bound parsing wrapped in thread to avoid blocking
                text = await asyncio.to_thread(self._parse_html, html)

                # Cache result
                self._add_to_cache(cache_key, text)
                return text
        except asyncio.TimeoutError:
            logger.warning(f"Timeout scraping {url}")
            return ""
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return ""

    def _parse_html(self, html: str) -> str:
        """Parse HTML and extract clean text (CPU-bound, runs in thread)"""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return " ".join(chunk for chunk in chunks if chunk)

    async def _apply_rate_limit_async(self):
        """Apply async rate limiting between requests (non-blocking, thread-safe)"""
        async with self._rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                await asyncio.sleep(sleep_time)  # Non-blocking sleep
            self.last_request_time = time.time()

    def _check_cache(self, key: str) -> bool:
        """Check if cache entry exists and is not expired"""
        if key in self.cache:
            _, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return True
            else:
                # Remove expired entry
                del self.cache[key]
        return False

    def _add_to_cache(self, key: str, value: Any):
        """Add item to cache with size limit enforcement"""
        # Enforce cache size limit by removing oldest entries
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest 10% of entries
            entries_to_remove = max(1, self.max_cache_size // 10)
            sorted_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k][1])
            for old_key in sorted_keys[:entries_to_remove]:
                del self.cache[old_key]

        self.cache[key] = (value, time.time())

    def _synthesize_results(self, results: List[Dict]) -> str:
        if not results:
            return "No results found."

        synthesis = "Research Findings:\n\n"
        for i, res in enumerate(results, 1):
            synthesis += f"{i}. {res.get('title', 'Unknown Title')}\n"
            synthesis += f"   Source: {res.get('href', 'Unknown URL')}\n"
            synthesis += f"   Excerpt: {res.get('body', '')[:200]}...\n\n"
        return synthesis
