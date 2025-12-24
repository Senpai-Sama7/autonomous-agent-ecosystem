"""
Model Manager - Fetch available models from different AI providers
Supports OpenAI, OpenRouter, Ollama, and Google Gemini
"""

import os
import requests
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about an AI model"""

    id: str
    name: str
    provider: str
    description: str = ""
    size: str = ""
    downloaded: bool = True
    context_window: int = 0
    supports_vision: bool = False
    supports_tools: bool = False


class ModelManager:
    """Manage AI models across different providers"""

    # Popular models for each provider (fallback)
    DEFAULT_MODELS = {
        "openai": [
            ModelInfo(
                "gpt-4o",
                "GPT-4o (Recommended)",
                "openai",
                "Most capable model with vision",
                context_window=128000,
                supports_vision=True,
                supports_tools=True,
            ),
            ModelInfo(
                "gpt-4o-mini",
                "GPT-4o Mini (Fast)",
                "openai",
                "Fast and affordable",
                context_window=128000,
                supports_vision=True,
                supports_tools=True,
            ),
            ModelInfo(
                "gpt-4-turbo",
                "GPT-4 Turbo",
                "openai",
                "Previous flagship",
                context_window=128000,
                supports_vision=True,
                supports_tools=True,
            ),
            ModelInfo(
                "gpt-3.5-turbo",
                "GPT-3.5 Turbo (Budget)",
                "openai",
                "Good for simple tasks",
                context_window=16385,
                supports_tools=True,
            ),
            ModelInfo(
                "o1-preview",
                "O1 Preview (Reasoning)",
                "openai",
                "Advanced reasoning model",
                context_window=128000,
            ),
            ModelInfo(
                "o1-mini",
                "O1 Mini (Fast Reasoning)",
                "openai",
                "Fast reasoning model",
                context_window=128000,
            ),
        ],
        "gemini": [
            ModelInfo(
                "gemini-2.0-flash",
                "Gemini 2.0 Flash (Recommended)",
                "gemini",
                "Latest fast model with advanced reasoning",
                context_window=1000000,
                supports_vision=True,
                supports_tools=True,
            ),
            ModelInfo(
                "gemini-2.0-flash-thinking-exp",
                "Gemini 2.0 Flash Thinking",
                "gemini",
                "Experimental thinking model",
                context_window=32767,
                supports_vision=True,
            ),
            ModelInfo(
                "gemini-1.5-pro",
                "Gemini 1.5 Pro",
                "gemini",
                "High capability, long context",
                context_window=2000000,
                supports_vision=True,
                supports_tools=True,
            ),
            ModelInfo(
                "gemini-1.5-flash",
                "Gemini 1.5 Flash",
                "gemini",
                "Fast responses, good for coding",
                context_window=1000000,
                supports_vision=True,
                supports_tools=True,
            ),
            ModelInfo(
                "gemini-1.5-flash-8b",
                "Gemini 1.5 Flash 8B",
                "gemini",
                "Lightweight, very fast",
                context_window=1000000,
                supports_vision=True,
            ),
        ],
        "openrouter": [
            ModelInfo(
                "anthropic/claude-3.5-sonnet",
                "Claude 3.5 Sonnet",
                "openrouter",
                "Best for coding",
                context_window=200000,
                supports_vision=True,
                supports_tools=True,
            ),
            ModelInfo(
                "anthropic/claude-3-opus",
                "Claude 3 Opus",
                "openrouter",
                "Most capable Claude",
                context_window=200000,
                supports_vision=True,
                supports_tools=True,
            ),
            ModelInfo(
                "openai/gpt-4o",
                "GPT-4o",
                "openrouter",
                "OpenAI's flagship",
                context_window=128000,
                supports_vision=True,
                supports_tools=True,
            ),
            ModelInfo(
                "google/gemini-pro-1.5",
                "Gemini Pro 1.5",
                "openrouter",
                "Google's best via OpenRouter",
                context_window=1000000,
                supports_vision=True,
            ),
            ModelInfo(
                "meta-llama/llama-3.1-405b-instruct",
                "Llama 3.1 405B",
                "openrouter",
                "Largest open model",
                context_window=131072,
                supports_tools=True,
            ),
            ModelInfo(
                "meta-llama/llama-3.1-70b-instruct",
                "Llama 3.1 70B",
                "openrouter",
                "Great open source model",
                context_window=131072,
                supports_tools=True,
            ),
            ModelInfo(
                "mistralai/mistral-large",
                "Mistral Large",
                "openrouter",
                "European AI leader",
                context_window=128000,
                supports_tools=True,
            ),
            ModelInfo(
                "deepseek/deepseek-chat",
                "DeepSeek Chat",
                "openrouter",
                "Strong coding model",
                context_window=64000,
            ),
        ],
        "ollama": [
            ModelInfo(
                "llama3.2",
                "Llama 3.2 (3B)",
                "ollama",
                "Fast, lightweight",
                "2GB",
                context_window=128000,
            ),
            ModelInfo(
                "llama3.2:1b",
                "Llama 3.2 (1B)",
                "ollama",
                "Ultra lightweight",
                "1.3GB",
                context_window=128000,
            ),
            ModelInfo(
                "llama3.1",
                "Llama 3.1 (8B)",
                "ollama",
                "Good balance",
                "4.7GB",
                context_window=128000,
            ),
            ModelInfo(
                "llama3.1:70b",
                "Llama 3.1 (70B)",
                "ollama",
                "Most capable local",
                "40GB",
                context_window=128000,
            ),
            ModelInfo(
                "mistral",
                "Mistral (7B)",
                "ollama",
                "Great for coding",
                "4.1GB",
                context_window=32000,
            ),
            ModelInfo(
                "codellama",
                "Code Llama (7B)",
                "ollama",
                "Specialized for code",
                "3.8GB",
                context_window=16000,
            ),
            ModelInfo(
                "codellama:34b",
                "Code Llama (34B)",
                "ollama",
                "Large code model",
                "19GB",
                context_window=16000,
            ),
            ModelInfo(
                "deepseek-coder:6.7b",
                "DeepSeek Coder (6.7B)",
                "ollama",
                "Excellent for coding",
                "3.8GB",
                context_window=16000,
            ),
            ModelInfo(
                "phi3",
                "Phi-3 (3.8B)",
                "ollama",
                "Microsoft's compact model",
                "2.3GB",
                context_window=128000,
            ),
            ModelInfo(
                "gemma2",
                "Gemma 2 (9B)",
                "ollama",
                "Google's open model",
                "5.4GB",
                context_window=8192,
            ),
            ModelInfo(
                "qwen2.5:7b",
                "Qwen 2.5 (7B)",
                "ollama",
                "Strong multilingual model",
                "4.4GB",
                context_window=128000,
            ),
            ModelInfo(
                "qwen2.5-coder:7b",
                "Qwen 2.5 Coder (7B)",
                "ollama",
                "Excellent code model",
                "4.4GB",
                context_window=128000,
            ),
        ],
    }

    @classmethod
    def get_ollama_models(
        cls, base_url: str = "http://localhost:11434"
    ) -> Tuple[List[ModelInfo], bool]:
        """
        Get list of downloaded Ollama models
        Returns: (list of models, is_ollama_running)
        """
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=3)
            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get("models", []):
                    name = model.get("name", "")
                    size_bytes = model.get("size", 0)
                    size_gb = f"{size_bytes / (1024**3):.1f}GB" if size_bytes else ""

                    # Clean up name for display
                    display_name = name.split(":")[0].title()
                    if ":" in name:
                        variant = name.split(":")[1]
                        display_name = f"{display_name} ({variant})"

                    models.append(
                        ModelInfo(
                            id=name,
                            name=display_name,
                            provider="ollama",
                            description="Downloaded",
                            size=size_gb,
                            downloaded=True,
                        )
                    )
                return models, True
            return [], False
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not running")
            return [], False
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {e}")
            return [], False

    @classmethod
    def get_available_ollama_models(cls) -> List[ModelInfo]:
        """Get list of popular Ollama models that can be downloaded"""
        return cls.DEFAULT_MODELS["ollama"]

    @classmethod
    def pull_ollama_model(
        cls,
        model_name: str,
        base_url: str = "http://localhost:11434",
        progress_callback=None,
    ) -> bool:
        """
        Download an Ollama model
        Returns True if successful
        """
        try:
            response = requests.post(
                f"{base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600,  # 10 minute timeout for large models
            )

            if response.status_code == 200:
                total_size = 0
                downloaded = 0

                for line in response.iter_lines():
                    if line:
                        import json

                        data = json.loads(line)

                        if "total" in data:
                            total_size = data["total"]
                        if "completed" in data:
                            downloaded = data["completed"]

                        if progress_callback and total_size > 0:
                            progress = downloaded / total_size
                            status = data.get("status", "Downloading...")
                            progress_callback(progress, status)

                        if data.get("status") == "success":
                            return True

                return True
            return False
        except Exception as e:
            logger.error(f"Error pulling Ollama model: {e}")
            return False

    @classmethod
    def check_ollama_status(cls, base_url: str = "http://localhost:11434") -> Dict:
        """Check if Ollama is running and get status"""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                data = response.json()
                return {
                    "running": True,
                    "model_count": len(data.get("models", [])),
                    "models": [m.get("name") for m in data.get("models", [])],
                }
        except Exception:
            pass
        return {"running": False, "model_count": 0, "models": []}

    @classmethod
    def check_gemini_status(cls, api_key: Optional[str] = None) -> Dict:
        """Check if Gemini API is accessible"""
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            return {"available": False, "error": "No API key configured"}

        try:
            # Try to list models to verify API key
            response = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": key},
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "").split("/")[-1] for m in data.get("models", [])]
                return {"available": True, "model_count": len(models), "models": models}
            else:
                return {"available": False, "error": f"API error: {response.status_code}"}
        except Exception as e:
            return {"available": False, "error": str(e)}

    @classmethod
    def get_gemini_models(cls, api_key: Optional[str] = None) -> List[ModelInfo]:
        """Get list of available Gemini models from API"""
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            return cls.DEFAULT_MODELS.get("gemini", [])

        try:
            response = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": key},
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                models = []
                for model_data in data.get("models", []):
                    name = model_data.get("name", "").split("/")[-1]
                    display_name = model_data.get("displayName", name)
                    description = model_data.get("description", "")

                    # Only include generateContent capable models
                    supported_methods = model_data.get("supportedGenerationMethods", [])
                    if "generateContent" not in supported_methods:
                        continue

                    # Parse capabilities
                    input_token_limit = model_data.get("inputTokenLimit", 0)
                    output_token_limit = model_data.get("outputTokenLimit", 0)

                    models.append(
                        ModelInfo(
                            id=name,
                            name=display_name,
                            provider="gemini",
                            description=description[:100] if description else "",
                            context_window=input_token_limit,
                            supports_vision="vision" in name.lower()
                            or "flash" in name.lower()
                            or "pro" in name.lower(),
                            supports_tools=True,
                        )
                    )

                # Sort by preference (2.0 first, then 1.5 pro, then 1.5 flash)
                def sort_key(m):
                    if "2.0" in m.id:
                        return 0
                    if "1.5-pro" in m.id:
                        return 1
                    if "1.5-flash" in m.id and "8b" not in m.id:
                        return 2
                    return 3

                models.sort(key=sort_key)
                return models if models else cls.DEFAULT_MODELS.get("gemini", [])
            return cls.DEFAULT_MODELS.get("gemini", [])
        except Exception as e:
            logger.error(f"Error fetching Gemini models: {e}")
            return cls.DEFAULT_MODELS.get("gemini", [])

    @classmethod
    def get_models_for_provider(
        cls, provider: str, api_key: str = None, base_url: str = None
    ) -> List[ModelInfo]:
        """Get available models for a provider"""

        if provider == "ollama":
            # Try to get downloaded models first
            models, is_running = cls.get_ollama_models(
                base_url or "http://localhost:11434"
            )
            if models:
                return models
            # Return downloadable models if none downloaded
            return cls.DEFAULT_MODELS.get("ollama", [])

        elif provider == "gemini":
            # Try to fetch from API
            models = cls.get_gemini_models(api_key)
            return models if models else cls.DEFAULT_MODELS.get("gemini", [])

        elif provider == "openai":
            # Could fetch from API, but use defaults for simplicity
            return cls.DEFAULT_MODELS.get("openai", [])

        elif provider == "openrouter":
            return cls.DEFAULT_MODELS.get("openrouter", [])

        return []

    @classmethod
    def get_model_by_id(cls, provider: str, model_id: str) -> Optional[ModelInfo]:
        """Get model info by ID"""
        models = cls.get_models_for_provider(provider)
        for model in models:
            if model.id == model_id:
                return model
        return None

    @classmethod
    def get_default_model(cls, provider: str) -> str:
        """Get default model ID for a provider"""
        defaults = {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.0-flash",
            "openrouter": "anthropic/claude-3.5-sonnet",
            "ollama": "llama3.2",
        }
        return defaults.get(provider, "gpt-4o-mini")

    @classmethod
    def get_all_providers(cls) -> List[Dict]:
        """Get list of all supported providers with metadata"""
        return [
            {
                "id": "openai",
                "name": "OpenAI",
                "description": "GPT-4o, GPT-4, GPT-3.5 - Industry standard",
                "requires_key": True,
                "key_env": "OPENAI_API_KEY",
                "free_tier": False,
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "description": "Gemini 2.0, 1.5 Pro/Flash - Free tier available",
                "requires_key": True,
                "key_env": "GEMINI_API_KEY",
                "free_tier": True,
            },
            {
                "id": "ollama",
                "name": "Ollama (Local)",
                "description": "Run models locally - Free, private",
                "requires_key": False,
                "key_env": None,
                "free_tier": True,
            },
            {
                "id": "openrouter",
                "name": "OpenRouter",
                "description": "Access many models with one key",
                "requires_key": True,
                "key_env": "OPENROUTER_API_KEY",
                "free_tier": True,
            },
        ]
