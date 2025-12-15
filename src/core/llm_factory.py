"""
LLM Client Factory
Centralizes the creation of LLM clients for OpenAI, OpenRouter, and Ollama.
Supports both Async and Sync clients.
"""

import os
import logging
from typing import Optional, Any

logger = logging.getLogger("LLMFactory")

try:
    from openai import OpenAI, AsyncOpenAI

    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    AsyncOpenAI = None
    HAS_OPENAI = False
    logger.warning("OpenAI library not found. LLM features will be disabled.")


class LLMFactory:
    @staticmethod
    def create_client(
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create and return an Async LLM client (AsyncOpenAI) based on the provider.
        Preferred for this async ecosystem.
        """
        if not HAS_OPENAI:
            return None

        provider = provider.lower()
        key, url = LLMFactory._get_config(provider, api_key, base_url)

        if not key and provider != "ollama":
            logger.warning(f"No API key found for provider {provider}")
            return None

        try:
            client = AsyncOpenAI(api_key=key, base_url=url)
            logger.info(f"Initialized Async LLM client for {provider}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Async LLM client: {e}")
            return None

    @staticmethod
    def create_sync_client(
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create and return a Sync LLM client (OpenAI).
        Use only if absolutely necessary (e.g. legacy code).
        """
        if not HAS_OPENAI:
            return None

        provider = provider.lower()
        key, url = LLMFactory._get_config(provider, api_key, base_url)

        if not key and provider != "ollama":
            logger.warning(f"No API key found for provider {provider}")
            return None

        try:
            client = OpenAI(api_key=key, base_url=url)
            logger.info(f"Initialized Sync LLM client for {provider}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Sync LLM client: {e}")
            return None

    @staticmethod
    def _get_config(
        provider: str, api_key: Optional[str], base_url: Optional[str]
    ) -> tuple:
        if provider == "openai":
            return api_key or os.getenv("OPENAI_API_KEY"), base_url

        elif provider == "openrouter":
            return (
                api_key or os.getenv("OPENROUTER_API_KEY"),
                base_url or "https://openrouter.ai/api/v1",
            )

        elif provider == "ollama":
            return "ollama", base_url or "http://localhost:11434/v1"

        else:
            logger.warning(f"Unknown provider {provider}, defaulting to OpenAI")
            return api_key or os.getenv("OPENAI_API_KEY"), base_url
