"""
Research Agent with Real Web Search Capabilities
"""
import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from .base_agent import BaseAgent, AgentCapability, AgentContext, TaskResult, AgentState

logger = logging.getLogger("ResearchAgent")

class ResearchAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, [AgentCapability.DATA_PROCESSING], config)
        self.ddgs = DDGS()
        self.max_results = config.get('max_search_results', 5)
        
        # Rate limiting
        self.min_request_interval = config.get('min_request_interval', 1.0)  # seconds between requests
        self.last_request_time = 0
        
        # Simple in-memory cache
        self.cache = {}
        self.cache_ttl = config.get('cache_ttl', 3600)  # 1 hour default

    async def execute_task(self, task: Dict[str, Any], context: AgentContext) -> TaskResult:
        try:
            self.state = AgentState.BUSY
            query = task.get('payload', {}).get('query') or task.get('description')
            
            if not query:
                return TaskResult(success=False, error_message="No query provided")

            logger.info(f"Performing real web search for: {query}")
            
            # 1. Real Web Search
            results = self._search_web(query)
            
            # 2. Content Extraction (Scraping)
            detailed_results = []
            for res in results[:3]:  # Limit scraping to top 3 to save time/bandwidth
                content = self._scrape_content(res['href'])
                if content:
                    res['full_content'] = content[:2000]  # Truncate for context window
                    detailed_results.append(res)

            # 3. Synthesis (Simple aggregation for now, could use LLM if configured)
            summary = self._synthesize_results(detailed_results)

            return TaskResult(
                success=True,
                result_data={
                    'summary': summary,
                    'sources': detailed_results,
                    'raw_search_results': results
                },
                execution_time=0.0  # Calculated by caller
            )

        except Exception as e:
            logger.error(f"Research failed: {e}")
            return TaskResult(success=False, error_message=str(e))
        finally:
            self.state = AgentState.ACTIVE

    def _search_web(self, query: str) -> List[Dict]:
        """Execute real DuckDuckGo search with caching"""
        # Check cache first
        cache_key = f"search:{query}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.info(f"Using cached search results for: {query}")
                return cached_data
        
        # Rate limiting
        self._apply_rate_limit()
        
        try:
            results = list(self.ddgs.text(query, max_results=self.max_results))
            # Cache results
            self.cache[cache_key] = (results, time.time())
            return results
        except Exception as e:
            logger.error(f"Search API error: {e}")
            return []

    def _scrape_content(self, url: str) -> str:
        """Scrape text content from URL with caching"""
        # Check cache
        cache_key = f"scrape:{url}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.info(f"Using cached content for: {url}")
                return cached_data
        
        # Rate limiting
        self._apply_rate_limit()
        
        try:
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            
            # Clean up
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Cache result
            self.cache[cache_key] = (text, time.time())
            return text
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return ""
    
    def _apply_rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _synthesize_results(self, results: List[Dict]) -> str:
        if not results:
            return "No results found."
        
        synthesis = "Research Findings:\n\n"
        for i, res in enumerate(results, 1):
            synthesis += f"{i}. {res.get('title', 'Unknown Title')}\n"
            synthesis += f"   Source: {res.get('href', 'Unknown URL')}\n"
            synthesis += f"   Excerpt: {res.get('body', '')[:200]}...\n\n"
        return synthesis