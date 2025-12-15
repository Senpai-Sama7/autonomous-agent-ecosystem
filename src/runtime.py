"""Runtime orchestration for the Autonomous Agent Ecosystem."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol

from src.agents.code_agent import CodeAgent
from src.agents.filesystem_agent import FileSystemAgent
from src.core.engine import AgentConfig, AgentEngine, Task, Workflow, WorkflowPriority
from src.core.nl_interface import NaturalLanguageInterface
from src.utils.logger import get_logger

logger = get_logger("Runtime")


class LLMFactoryProtocol(Protocol):
    """Protocol to allow injecting different LLM factories in tests."""

    def create_client(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Any: ...


class ConfigLoaderProtocol(Protocol):
    """Protocol for loading configuration."""

    def load_configs(self) -> Dict[str, Any]: ...


class DashboardProtocol(Protocol):
    """Protocol for interacting with the monitoring dashboard."""

    async def start_monitoring(self) -> None: ...

    def register_agent(self, agent_id: str, metadata: Dict[str, Any]) -> None: ...

    def get_system_health(self) -> Mapping[str, Any]: ...


@dataclass
class EcosystemDependencies:
    """Dependencies required by the ecosystem runtime."""

    engine: AgentEngine
    config_loader: ConfigLoaderProtocol
    llm_factory: LLMFactoryProtocol
    dashboard: DashboardProtocol


class AutonomousAgentEcosystem:
    """Main application class that orchestrates the autonomous agent ecosystem."""

    def __init__(
        self,
        config: Mapping[str, Any],
        dependencies: EcosystemDependencies,
        loaded_configs: Optional[Dict[str, Any]] = None,
    ):
        self.config: Dict[str, Any] = dict(config)
        self.engine = dependencies.engine
        self.dashboard = dependencies.dashboard
        self.config_loader = dependencies.config_loader
        self.llm_factory = dependencies.llm_factory
        self.agents: Dict[str, Any] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.loaded_configs: Dict[str, Any] = (
            loaded_configs or self.config_loader.load_configs()
        )
        logger.info("Autonomous Agent Ecosystem initialized")

    async def initialize_agents(self) -> None:
        """Initialize and register all agents."""

        logger.info("Initializing agents...")
        agent_configs = self.loaded_configs.get("agents", {})
        llm_config = self.loaded_configs.get("llm", {})

        provider = self.config.get("llm_provider") or getattr(
            llm_config, "provider", "openai"
        )
        api_key = self.config.get("api_key") or getattr(llm_config, "api_key", None)
        base_url = self.config.get("api_base") or getattr(llm_config, "api_base", None)
        model_name = self.config.get("model_name") or getattr(
            llm_config, "model_name", "gpt-3.5-turbo"
        )

        llm_client = self.llm_factory.create_client(
            provider=provider, api_key=api_key, base_url=base_url
        )

        research_agent = None

        code_config_dict = dict(agent_configs.get("code_agent_001", {}))
        code_config_dict.update(
            {
                "llm_provider": provider,
                "model_name": model_name,
                "api_key": api_key,
                "api_base": base_url,
            }
        )
        code_agent = CodeAgent(
            agent_id="code_agent_001", config=code_config_dict, llm_client=llm_client
        )

        fs_config_dict = dict(agent_configs.get("filesystem_agent_001", {}))
        fs_agent = FileSystemAgent(
            agent_id="filesystem_agent_001", config=fs_config_dict
        )

        await self.engine.register_agent(
            AgentConfig(
                agent_id="research_agent_001",
                capabilities=[
                    "web_search",
                    "content_extraction",
                    "knowledge_synthesis",
                    "data_processing",
                ],
                max_concurrent_tasks=2,
                reliability_score=0.92,
                cost_per_operation=1.5,
            ),
            instance=research_agent if research_agent else None,
        )

        await self.engine.register_agent(
            AgentConfig(
                agent_id="code_agent_001",
                capabilities=[
                    "code_generation",
                    "code_optimization",
                    "debugging",
                    "optimization",
                ],
                max_concurrent_tasks=code_config_dict.get("max_concurrent_tasks", 2),
                reliability_score=0.88,
                cost_per_operation=2.0,
            ),
            instance=code_agent,
        )

        await self.engine.register_agent(
            AgentConfig(
                agent_id="filesystem_agent_001",
                capabilities=["data_processing", "file_operations"],
                max_concurrent_tasks=fs_config_dict.get("max_concurrent_tasks", 5),
                reliability_score=0.99,
                cost_per_operation=0.5,
            ),
            instance=fs_agent,
        )

        self.agents["research_agent_001"] = research_agent
        self.agents["code_agent_001"] = code_agent
        self.agents["filesystem_agent_001"] = fs_agent

        self.dashboard.register_agent(
            "research_agent_001", {"state": "active", "capabilities": ["web_search"]}
        )
        self.dashboard.register_agent(
            "code_agent_001", {"state": "active", "capabilities": ["code_generation"]}
        )
        self.dashboard.register_agent(
            "filesystem_agent_001",
            {"state": "active", "capabilities": ["file_operations"]},
        )

        logger.info("Agents initialized and registered successfully")

    async def create_sample_workflows(self) -> None:
        """Create sample workflows from test YAML."""

        logger.info("Creating sample workflows from test config...")
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed; skipping sample workflows")
            return

        with open(
            "/home/donovan/Projects/AI/Astro/config/simple_test.yaml",
            "r",
            encoding="utf-8",
        ) as f:
            test_workflows = yaml.safe_load(f)

        for wf_name, wf_config in test_workflows.items():
            workflow = Workflow(
                workflow_id=f"wf_{wf_name}_{uuid.uuid4().hex[:8]}",
                name=wf_config["description"],
                tasks=[
                    Task(
                        task_id=f"{task['id']}_{uuid.uuid4().hex[:8]}",
                        description=task["instruction"],
                        required_capabilities=[task["capability"]],
                        payload=task.get("payload", {}),
                        priority=WorkflowPriority(wf_config["priority"]),
                    )
                    for task in wf_config["tasks"]
                ],
                priority=WorkflowPriority(wf_config["priority"]),
                max_execution_time=wf_config.get("max_execution_time", 3600.0),
                cost_budget=wf_config.get("budget", 100.0),
            )
            self.workflows[workflow.workflow_id] = workflow
            await self.engine.submit_workflow(workflow)
            logger.info(
                f"Submitted test workflow: {wf_name} ({wf_config['description']})"
            )

    async def start_system(self) -> None:
        """Start the autonomous agent ecosystem."""

        logger.info("🚀 Starting Autonomous Agent Ecosystem (PRODUCTION MODE)")
        logger.info("=" * 60)

        try:
            await self.initialize_agents()
            await self.create_sample_workflows()
            engine_task = asyncio.create_task(self.engine.start_engine())
            monitoring_task = asyncio.create_task(self.dashboard.start_monitoring())

            duration = self.config.get("duration", 120)
            logger.info(
                f"System running for {duration} seconds... Press Ctrl+C to stop."
            )
            logger.info("=" * 60)

            await asyncio.sleep(duration)

            engine_task.cancel()
            monitoring_task.cancel()
            try:
                await engine_task
                await monitoring_task
            except asyncio.CancelledError:
                pass

            logger.info("✅ System shutdown completed gracefully")

        except KeyboardInterrupt:
            logger.info("🛑 System interrupted by user")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ System error: {exc}", exc_info=True)
        finally:
            await self.generate_final_report()

    async def generate_final_report(self) -> None:
        """Generate a comprehensive final report of system execution."""

        logger.info("\n" + "=" * 60)
        logger.info("📊 FINAL SYSTEM REPORT")
        logger.info("=" * 60)

        system_health = self.dashboard.get_system_health()
        logger.info(
            f"📈 System Health Score: {system_health['health_score']:.2f} ({system_health['status']})"
        )
        logger.info(f"🤖 Active Agents: {system_health['active_agents_count']}")

        logger.info("\n" + "-" * 40)
        logger.info("_WORKFLOW EXECUTION SUMMARY_")
        logger.info("-" * 40)

        for workflow_id, workflow in self.workflows.items():
            completed_tasks = sum(
                1
                for task in workflow.tasks
                if task.task_id in self.engine.completed_tasks
            )
            total_tasks = len(workflow.tasks)
            status = (
                "✅ Completed"
                if completed_tasks == total_tasks
                else f"⚠️ {completed_tasks}/{total_tasks} Tasks"
            )
            logger.info(f"\n{workflow.name}: {status}")

        logger.info("\n" + "=" * 60)
        logger.info("✨ EXECUTION COMPLETE")
        logger.info("=" * 60)


async def interactive_mode(ecosystem: AutonomousAgentEcosystem) -> None:
    """Run the system in interactive mode."""

    try:
        await ecosystem.initialize_agents()
        engine_task = asyncio.create_task(ecosystem.engine.start_engine())
        monitor_task = asyncio.create_task(ecosystem.dashboard.start_monitoring())

        llm_config = ecosystem.loaded_configs.get("llm", {})
        llm_client = ecosystem.llm_factory.create_client(
            provider=ecosystem.config.get("llm_provider")
            or getattr(llm_config, "provider", "openai"),
            api_key=ecosystem.config.get("api_key")
            or getattr(llm_config, "api_key", None),
            base_url=ecosystem.config.get("api_base")
            or getattr(llm_config, "api_base", None),
        )

        nl_interface = NaturalLanguageInterface(
            engine=ecosystem.engine,
            llm_client=llm_client,
            model_name=ecosystem.config.get("model_name")
            or getattr(llm_config, "model_name", "gpt-3.5-turbo"),
        )

        print("\n💬 Interactive Mode Enabled. Type your request (or 'exit'):")
        while True:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, input, ">> "
            )
            if user_input.lower() in ["exit", "quit"]:
                break

            try:
                workflow_id = await nl_interface.process_request(user_input)
                print(f"✅ Workflow submitted: {workflow_id}")
            except Exception as exc:  # noqa: BLE001
                print(f"❌ Error: {exc}")

        engine_task.cancel()
        monitor_task.cancel()
        try:
            await engine_task
            await monitor_task
        except asyncio.CancelledError:
            pass

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await ecosystem.generate_final_report()
