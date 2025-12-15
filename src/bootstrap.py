"""Bootstrap utilities to construct and run the Astro runtime."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Mapping, Optional, Sequence

from src.cli import CLIOptions, parse_arguments
from src.core.engine import AgentEngine
from src.core.llm_factory import LLMFactory
from src.monitoring.monitoring_dashboard import MonitoringDashboard
from src.runtime import AutonomousAgentEcosystem, EcosystemDependencies
from src.utils.config_loader import ConfigLoader, LLMConfig
from src.utils.logger import configure_logging


class LLMFactoryAdapter:
    """Adapter to use the static :class:`LLMFactory` as an injectable dependency."""

    def create_client(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Any:
        return LLMFactory.create_client(
            provider=provider, api_key=api_key, base_url=base_url
        )


def merge_configurations(
    cli_options: CLIOptions, loaded_configs: Mapping[str, Any]
) -> Dict[str, Any]:
    """Merge CLI options with loaded configuration values."""

    llm_config: LLMConfig | Mapping[str, Any] = loaded_configs.get("llm", LLMConfig())

    return {
        "duration": cli_options.duration,
        "debug": cli_options.debug,
        "interactive": cli_options.interactive,
        "llm_provider": cli_options.llm_provider
        or getattr(llm_config, "provider", None),
        "model_name": cli_options.model_name or getattr(llm_config, "model_name", None),
        "api_key": cli_options.api_key or getattr(llm_config, "api_key", None),
        "api_base": cli_options.api_base or getattr(llm_config, "api_base", None),
    }


def build_ecosystem(
    cli_args: Optional[Sequence[str]] = None,
    *,
    config_loader: Optional[ConfigLoader] = None,
    llm_factory: Optional[LLMFactoryAdapter] = None,
    dashboard: Optional[MonitoringDashboard] = None,
    engine: Optional[AgentEngine] = None,
    loaded_configs: Optional[Dict[str, Any]] = None,
) -> AutonomousAgentEcosystem:
    """Construct the ecosystem with injectable dependencies."""

    cli_options = parse_arguments(cli_args)
    configure_logging(
        log_level="DEBUG" if cli_options.debug else "INFO", log_to_file=True
    )

    active_config_loader = config_loader or ConfigLoader()
    existing_configs = loaded_configs or active_config_loader.load_configs()
    runtime_config = merge_configurations(cli_options, existing_configs)

    dependencies = EcosystemDependencies(
        engine=engine or AgentEngine(),
        config_loader=active_config_loader,
        llm_factory=llm_factory or LLMFactoryAdapter(),
        dashboard=dashboard or MonitoringDashboard(),
    )

    return AutonomousAgentEcosystem(
        config=runtime_config,
        dependencies=dependencies,
        loaded_configs=existing_configs,
    )


def run_application(cli_args: Optional[Sequence[str]] = None) -> None:
    """Entrypoint used by ``src/main.py``."""

    ecosystem = build_ecosystem(cli_args)
    if ecosystem.config.get("interactive"):
        asyncio.run(interactive_entrypoint(ecosystem))
    else:
        asyncio.run(ecosystem.start_system())


async def interactive_entrypoint(ecosystem: AutonomousAgentEcosystem) -> None:
    """Launch the interactive runtime."""

    from src.runtime import interactive_mode

    await interactive_mode(ecosystem)
