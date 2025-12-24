"""
LLM Client Factory
Centralizes the creation of LLM clients for OpenAI, OpenRouter, Ollama, and Google Gemini.
Supports both Async and Sync clients with unified interface.
"""

import os
import logging
from typing import Optional, Any, AsyncIterator, Iterator
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("LLMFactory")


class LLMProvider(Enum):
    """Supported LLM providers"""

    OPENAI = "openai"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    GEMINI = "gemini"


@dataclass
class LLMResponse:
    """Unified response format for all providers"""

    content: str
    model: str
    provider: str
    usage: Optional[dict] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None


@dataclass
class StreamChunk:
    """Chunk for streaming responses"""

    content: str
    is_final: bool = False
    usage: Optional[dict] = None


# Check for OpenAI SDK
try:
    from openai import OpenAI, AsyncOpenAI

    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    AsyncOpenAI = None
    HAS_OPENAI = False
    logger.warning("OpenAI library not found. OpenAI/OpenRouter/Ollama features disabled.")

# Check for Google GenAI SDK
try:
    from google import genai
    from google.genai import types

    HAS_GEMINI = True
except ImportError:
    genai = None
    types = None
    HAS_GEMINI = False
    logger.warning("Google GenAI library not found. Gemini features disabled.")


class GeminiClient:
    """Wrapper for Google Gemini API with unified interface"""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        if not HAS_GEMINI:
            raise ImportError("google-genai package not installed")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.api_key = api_key

    async def chat_completion(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse | AsyncIterator[StreamChunk]:
        """Create a chat completion (async)"""
        # Convert OpenAI-style messages to Gemini format
        contents = self._convert_messages(messages)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        if stream:
            return self._stream_response(contents, config)
        else:
            response = await self.client.aio.models.generate_content(
                model=self.model, contents=contents, config=config
            )

            return LLMResponse(
                content=response.text if response.text else "",
                model=self.model,
                provider="gemini",
                usage=(
                    {
                        "prompt_tokens": response.usage_metadata.prompt_token_count
                        if response.usage_metadata
                        else 0,
                        "completion_tokens": response.usage_metadata.candidates_token_count
                        if response.usage_metadata
                        else 0,
                        "total_tokens": response.usage_metadata.total_token_count
                        if response.usage_metadata
                        else 0,
                    }
                    if response.usage_metadata
                    else None
                ),
                finish_reason=(
                    response.candidates[0].finish_reason.name
                    if response.candidates
                    else None
                ),
                raw_response=response,
            )

    async def _stream_response(
        self, contents: list, config: types.GenerateContentConfig
    ) -> AsyncIterator[StreamChunk]:
        """Stream response chunks"""
        async for chunk in self.client.aio.models.generate_content_stream(
            model=self.model, contents=contents, config=config
        ):
            if chunk.text:
                yield StreamChunk(content=chunk.text, is_final=False)

        yield StreamChunk(content="", is_final=True)

    def chat_completion_sync(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse | Iterator[StreamChunk]:
        """Create a chat completion (sync)"""
        contents = self._convert_messages(messages)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        if stream:
            return self._stream_response_sync(contents, config)
        else:
            response = self.client.models.generate_content(
                model=self.model, contents=contents, config=config
            )

            return LLMResponse(
                content=response.text if response.text else "",
                model=self.model,
                provider="gemini",
                usage=(
                    {
                        "prompt_tokens": response.usage_metadata.prompt_token_count
                        if response.usage_metadata
                        else 0,
                        "completion_tokens": response.usage_metadata.candidates_token_count
                        if response.usage_metadata
                        else 0,
                        "total_tokens": response.usage_metadata.total_token_count
                        if response.usage_metadata
                        else 0,
                    }
                    if response.usage_metadata
                    else None
                ),
                finish_reason=(
                    response.candidates[0].finish_reason.name
                    if response.candidates
                    else None
                ),
                raw_response=response,
            )

    def _stream_response_sync(
        self, contents: list, config: types.GenerateContentConfig
    ) -> Iterator[StreamChunk]:
        """Stream response chunks (sync)"""
        for chunk in self.client.models.generate_content_stream(
            model=self.model, contents=contents, config=config
        ):
            if chunk.text:
                yield StreamChunk(content=chunk.text, is_final=False)

        yield StreamChunk(content="", is_final=True)

    def _convert_messages(self, messages: list) -> list:
        """Convert OpenAI-style messages to Gemini format"""
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Gemini handles system prompts differently
                system_instruction = content
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=content)]))
            else:  # user
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))

        # Prepend system instruction to first user message if present
        if system_instruction and contents:
            first_content = contents[0]
            if first_content.parts:
                first_content.parts[0] = types.Part(
                    text=f"{system_instruction}\n\n{first_content.parts[0].text}"
                )

        return contents


class UnifiedLLMClient:
    """Unified LLM client that wraps all providers"""

    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = LLMProvider(provider.lower())
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client = None
        self._async_client = None

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate client based on provider"""
        if self.provider == LLMProvider.GEMINI:
            if not HAS_GEMINI:
                raise ImportError("google-genai package not installed")
            key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError("Gemini API key not provided")
            self._client = GeminiClient(api_key=key, model=self.model or "gemini-2.0-flash")

        elif self.provider in [LLMProvider.OPENAI, LLMProvider.OPENROUTER, LLMProvider.OLLAMA]:
            if not HAS_OPENAI:
                raise ImportError("openai package not installed")

            key, url = self._get_openai_config()
            self._async_client = AsyncOpenAI(api_key=key, base_url=url)
            self._client = OpenAI(api_key=key, base_url=url)

    def _get_openai_config(self) -> tuple:
        """Get API key and base URL for OpenAI-compatible providers"""
        if self.provider == LLMProvider.OPENAI:
            return (
                self.api_key or os.getenv("OPENAI_API_KEY"),
                self.base_url,
            )
        elif self.provider == LLMProvider.OPENROUTER:
            return (
                self.api_key or os.getenv("OPENROUTER_API_KEY"),
                self.base_url or "https://openrouter.ai/api/v1",
            )
        elif self.provider == LLMProvider.OLLAMA:
            return (
                "ollama",  # Ollama doesn't need a real key
                self.base_url or "http://localhost:11434/v1",
            )
        return None, None

    async def chat(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse | AsyncIterator[StreamChunk]:
        """Unified chat completion interface (async)"""
        use_model = model or self.model

        if self.provider == LLMProvider.GEMINI:
            self._client.model = use_model or "gemini-2.0-flash"
            return await self._client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs,
            )
        else:
            # OpenAI-compatible providers
            if stream:
                return self._stream_openai(messages, use_model, temperature, max_tokens, **kwargs)
            else:
                response = await self._async_client.chat.completions.create(
                    model=use_model or "gpt-4o-mini",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    provider=self.provider.value,
                    usage=(
                        {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }
                        if response.usage
                        else None
                    ),
                    finish_reason=response.choices[0].finish_reason,
                    raw_response=response,
                )

    async def _stream_openai(
        self,
        messages: list,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream responses from OpenAI-compatible APIs"""
        stream = await self._async_client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(content=chunk.choices[0].delta.content, is_final=False)

        yield StreamChunk(content="", is_final=True)

    def chat_sync(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse | Iterator[StreamChunk]:
        """Unified chat completion interface (sync)"""
        use_model = model or self.model

        if self.provider == LLMProvider.GEMINI:
            self._client.model = use_model or "gemini-2.0-flash"
            return self._client.chat_completion_sync(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs,
            )
        else:
            if stream:
                return self._stream_openai_sync(messages, use_model, temperature, max_tokens, **kwargs)
            else:
                response = self._client.chat.completions.create(
                    model=use_model or "gpt-4o-mini",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    provider=self.provider.value,
                    usage=(
                        {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }
                        if response.usage
                        else None
                    ),
                    finish_reason=response.choices[0].finish_reason,
                    raw_response=response,
                )

    def _stream_openai_sync(
        self,
        messages: list,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """Stream responses from OpenAI-compatible APIs (sync)"""
        stream = self._client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(content=chunk.choices[0].delta.content, is_final=False)

        yield StreamChunk(content="", is_final=True)


class LLMFactory:
    """Factory for creating LLM clients - maintains backward compatibility"""

    @staticmethod
    def create_client(
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create and return an Async LLM client based on the provider.
        For backward compatibility, returns AsyncOpenAI for OpenAI-compatible providers.
        For Gemini, returns the unified client.
        """
        provider = provider.lower()

        if provider == "gemini":
            try:
                return UnifiedLLMClient(
                    provider=provider, api_key=api_key, base_url=base_url, model=model
                )
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                return None

        # For backward compatibility, return AsyncOpenAI directly
        if not HAS_OPENAI:
            return None

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
        model: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create and return a Sync LLM client.
        """
        provider = provider.lower()

        if provider == "gemini":
            try:
                return UnifiedLLMClient(
                    provider=provider, api_key=api_key, base_url=base_url, model=model
                )
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                return None

        if not HAS_OPENAI:
            return None

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
    def create_unified_client(
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[UnifiedLLMClient]:
        """
        Create a unified LLM client with consistent interface across all providers.
        This is the recommended way to create clients for new code.
        """
        try:
            return UnifiedLLMClient(
                provider=provider, api_key=api_key, base_url=base_url, model=model
            )
        except Exception as e:
            logger.error(f"Failed to create unified client: {e}")
            return None

    @staticmethod
    def _get_config(provider: str, api_key: Optional[str], base_url: Optional[str]) -> tuple:
        """Get API configuration for a provider"""
        if provider == "openai":
            return api_key or os.getenv("OPENAI_API_KEY"), base_url

        elif provider == "openrouter":
            return (
                api_key or os.getenv("OPENROUTER_API_KEY"),
                base_url or "https://openrouter.ai/api/v1",
            )

        elif provider == "ollama":
            return "ollama", base_url or "http://localhost:11434/v1"

        elif provider == "gemini":
            return (
                api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
                base_url or "https://generativelanguage.googleapis.com/v1beta/openai/",
            )

        else:
            logger.warning(f"Unknown provider {provider}, defaulting to OpenAI")
            return api_key or os.getenv("OPENAI_API_KEY"), base_url

    @staticmethod
    def get_available_providers() -> list:
        """Get list of available providers based on installed packages"""
        providers = []
        if HAS_OPENAI:
            providers.extend(["openai", "openrouter", "ollama"])
        if HAS_GEMINI:
            providers.append("gemini")
        return providers
