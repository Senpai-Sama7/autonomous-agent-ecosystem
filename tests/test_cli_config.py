from __future__ import annotations

from src.bootstrap import merge_configurations
from src.cli import CLIOptions, parse_arguments
from src.utils.config_loader import LLMConfig


class TestCLIParsing:
    def test_parse_arguments_defaults(self) -> None:
        options = parse_arguments([])

        assert options.duration == 120
        assert options.debug is False
        assert options.interactive is False
        assert options.llm_provider is None
        assert options.model_name is None
        assert options.api_key is None
        assert options.api_base is None

    def test_parse_arguments_overrides(self) -> None:
        options = parse_arguments(
            [
                "--duration",
                "10",
                "--debug",
                "--interactive",
                "--llm-provider",
                "openrouter",
                "--model-name",
                "custom-model",
                "--api-key",
                "secret",
                "--api-base",
                "https://router",
            ]
        )

        assert options.duration == 10
        assert options.debug is True
        assert options.interactive is True
        assert options.llm_provider == "openrouter"
        assert options.model_name == "custom-model"
        assert options.api_key == "secret"
        assert options.api_base == "https://router"


class TestConfigurationMerging:
    def test_merge_cli_overrides_loader(self) -> None:
        cli_options = CLIOptions(
            duration=60,
            debug=True,
            interactive=False,
            llm_provider="openrouter",
            model_name="gpt-best",
            api_key="cli-key",
            api_base="https://cli",
        )
        loaded_configs = {
            "llm": LLMConfig(
                provider="openai",
                model_name="file",
                api_key="file-key",
                api_base="https://file",
            )
        }

        merged = merge_configurations(cli_options, loaded_configs)

        assert merged["duration"] == 60
        assert merged["llm_provider"] == "openrouter"
        assert merged["model_name"] == "gpt-best"
        assert merged["api_key"] == "cli-key"
        assert merged["api_base"] == "https://cli"

    def test_merge_respects_loader_defaults(self) -> None:
        cli_options = CLIOptions(duration=30, debug=False, interactive=True)
        loaded_configs = {
            "llm": LLMConfig(
                provider="ollama",
                model_name="loader-model",
                api_key="loader-key",
                api_base="https://loader",
            )
        }

        merged = merge_configurations(cli_options, loaded_configs)

        assert merged["duration"] == 30
        assert merged["interactive"] is True
        assert merged["llm_provider"] == "ollama"
        assert merged["model_name"] == "loader-model"
        assert merged["api_key"] == "loader-key"
        assert merged["api_base"] == "https://loader"
