"""
LLM Client Factory
Centralizes the creation of LLM clients for OpenAI, OpenRouter, and Ollama.
"""
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("LLMFactory")

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logger.warning("OpenAI library not found. LLM features will be disabled.")

class LLMFactory:
    @staticmethod
    def create_client(provider: str = "openai", 
                      api_key: Optional[str] = None, 
                      base_url: Optional[str] = None) -> Optional[Any]:
        """
        Create and return an LLM client based on the provider.
        """
        if not OpenAI:
            return None

        provider = provider.lower()
        
        # Determine API Key and Base URL based on provider
        if provider == "openai":
            final_api_key = api_key or os.getenv("OPENAI_API_KEY")
            final_base_url = base_url # Default is None (standard OpenAI)
            
        elif provider == "openrouter":
            final_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            final_base_url = base_url or "https://openrouter.ai/api/v1"
            
        elif provider == "ollama":
            final_api_key = "ollama" # Dummy key required
            final_base_url = base_url or "http://localhost:11434/v1"
            
        else:
            logger.warning(f"Unknown provider {provider}, defaulting to OpenAI")
            final_api_key = api_key or os.getenv("OPENAI_API_KEY")
            final_base_url = base_url

        if not final_api_key and provider != "ollama":
            logger.warning(f"No API key found for provider {provider}")
            return None

        try:
            client = OpenAI(api_key=final_api_key, base_url=final_base_url)
            logger.info(f"Initialized LLM client for {provider}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            return None
