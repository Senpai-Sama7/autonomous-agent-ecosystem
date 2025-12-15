"""
Model Manager - Fetch available models from different AI providers
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


class ModelManager:
    """Manage AI models across different providers"""

    # Popular models for each provider (fallback)
    DEFAULT_MODELS = {
        "openai": [
            ModelInfo("gpt-4o", "GPT-4o (Recommended)", "openai", "Most capable model"),
            ModelInfo(
                "gpt-4o-mini", "GPT-4o Mini (Fast)", "openai", "Fast and affordable"
            ),
            ModelInfo("gpt-4-turbo", "GPT-4 Turbo", "openai", "Previous flagship"),
            ModelInfo(
                "gpt-3.5-turbo",
                "GPT-3.5 Turbo (Budget)",
                "openai",
                "Good for simple tasks",
            ),
        ],
        "openrouter": [
            ModelInfo(
                "anthropic/claude-3.5-sonnet",
                "Claude 3.5 Sonnet",
                "openrouter",
                "Best for coding",
            ),
            ModelInfo("openai/gpt-4o", "GPT-4o", "openrouter", "OpenAI's flagship"),
            ModelInfo(
                "google/gemini-pro-1.5", "Gemini Pro 1.5", "openrouter", "Google's best"
            ),
            ModelInfo(
                "meta-llama/llama-3.1-70b-instruct",
                "Llama 3.1 70B",
                "openrouter",
                "Open source",
            ),
            ModelInfo(
                "mistralai/mistral-large", "Mistral Large", "openrouter", "European AI"
            ),
        ],
        "ollama": [
            ModelInfo(
                "llama3.2", "Llama 3.2 (3B)", "ollama", "Fast, lightweight", "2GB"
            ),
            ModelInfo("llama3.1", "Llama 3.1 (8B)", "ollama", "Good balance", "4.7GB"),
            ModelInfo("mistral", "Mistral (7B)", "ollama", "Great for coding", "4.1GB"),
            ModelInfo(
                "codellama",
                "Code Llama (7B)",
                "ollama",
                "Specialized for code",
                "3.8GB",
            ),
            ModelInfo(
                "phi3", "Phi-3 (3.8B)", "ollama", "Microsoft's compact model", "2.3GB"
            ),
            ModelInfo(
                "gemma2", "Gemma 2 (9B)", "ollama", "Google's open model", "5.4GB"
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
        except:
            pass
        return {"running": False, "model_count": 0, "models": []}

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

        elif provider == "openai":
            # Could fetch from API, but use defaults for simplicity
            return cls.DEFAULT_MODELS.get("openai", [])

        elif provider == "openrouter":
            return cls.DEFAULT_MODELS.get("openrouter", [])

        return []
