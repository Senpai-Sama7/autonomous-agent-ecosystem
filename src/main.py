"""
Autonomous Agent Ecosystem - Main Application
Updated by C0Di3 to support Declarative Workflows and Templating.
"""

import asyncio
import json
import uuid
from typing import Dict, Any, List
import argparse
import os

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import configure_logging, get_logger
from core.engine import AgentEngine, AgentConfig, Workflow, Task, WorkflowPriority
from core.nl_interface import NaturalLanguageInterface

# from agents.research_agent import ResearchAgent  # Temporarily disabled
from agents.base_agent import TaskResult
from agents.code_agent import CodeAgent
from agents.filesystem_agent import FileSystemAgent
from monitoring.monitoring_dashboard import MonitoringDashboard
from core.llm_factory import LLMFactory
from utils.config_loader import ConfigLoader

# Logger will be configured in main() based on CLI args
logger = get_logger("MainApplication")


class UnavailableAgent:
    """Stub agent used when an implementation is unavailable."""

    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.is_available = False

    async def execute_task(self, task: Dict[str, Any] | None = None, context: Any | None = None) -> TaskResult:
        return TaskResult(
            success=False,
            error_message=f"Agent {self.agent_id} unavailable",
            result_data={"agent_id": self.agent_id, "available": False},
        )


class AutonomousAgentEcosystem:
    """
    Main application class that orchestrates the entire autonomous agent ecosystem.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        if config is None:
            config = {}
        """Initialize the ecosystem"""
        self.engine = AgentEngine()
        self.dashboard = MonitoringDashboard()
        self.agents = {}
        self.workflows = {}

        # 1. Initialize ConfigLoader and load file-based configs
        self.config_loader = ConfigLoader()
        self.loaded_configs = self.config_loader.load_configs()

        # 2. Merge CLI config (overrides files)
        self.config = config or {}

        logger.info("Autonomous Agent Ecosystem initialized")

    async def initialize_agents(self):
        """Initialize and register all agents"""
        logger.info("Initializing agents...")

        # Get agent configs from loader
        agent_configs = self.loaded_configs.get("agents", {})

        # Determine LLM settings: CLI > Config File > Default
        llm_config = self.loaded_configs.get("llm", {})

        provider = self.config.get("llm_provider") or getattr(
            llm_config, "provider", "openai"
        )
        api_key = self.config.get("api_key") or getattr(llm_config, "api_key", None)
        base_url = self.config.get("api_base") or getattr(llm_config, "api_base", None)
        model_name = self.config.get("model_name") or getattr(
            llm_config, "model_name", "gpt-3.5-turbo"
        )

        # Create shared LLM client
        llm_client = LLMFactory.create_client(
            provider=provider, api_key=api_key, base_url=base_url
        )

        # --- Research Agent ---
        # research_config_dict = agent_configs.get("research_agent_001", {})
        # research_agent = ResearchAgent(
        #     agent_id="research_agent_001", config=research_config_dict
        # )
        research_agent = None  # Temporarily disabled

        # --- Code Agent ---
        code_config_dict = agent_configs.get("code_agent_001", {})
        # Update with runtime LLM settings
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

        # --- FileSystem Agent ---
        fs_config_dict = agent_configs.get("filesystem_agent_001", {})
        fs_agent = FileSystemAgent(
            agent_id="filesystem_agent_001", config=fs_config_dict
        )

        research_config = AgentConfig(
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
        )

        code_config = AgentConfig(
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
        )

        filesystem_config = AgentConfig(
            agent_id="filesystem_agent_001",
            capabilities=["data_processing", "file_operations"],
            max_concurrent_tasks=fs_config_dict.get("max_concurrent_tasks", 5),
            reliability_score=0.99,
            cost_per_operation=0.5,
        )

        # Register agents with the engine, substituting stubs when needed
        await self._safe_register_agent(research_config, research_agent)
        await self._safe_register_agent(code_config, code_agent)
        await self._safe_register_agent(filesystem_config, fs_agent)

    async def _safe_register_agent(self, config: AgentConfig, instance: Any | None) -> None:
        if instance is None:
            logger.warning(
                "Agent implementation unavailable; registering stub", extra={"agent_id": config.agent_id}
            )
            instance = UnavailableAgent(config.agent_id, config.capabilities)

        await self.engine.register_agent(config, instance=instance)
        self.agents[config.agent_id] = instance

        # Register with dashboard
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

    async def create_sample_workflows(self):
        """Create sample workflows from YAML configuration."""

        logger.info("Creating sample workflows from config...")

        workflow_path_override = self.config.get("sample_workflows")
        test_workflows = self.config_loader.load_workflows(
            workflow_path=workflow_path_override
        )

        if not test_workflows:
            logger.info("No sample workflows loaded; skipping submission.")
            return

        for wf_name, wf_config in test_workflows.items():
            if not isinstance(wf_config, dict):
                logger.warning("Skipping workflow '%s' because it is not a mapping", wf_name)
                continue

            try:
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
            except KeyError as exc:
                logger.error(
                    "Workflow '%s' missing required field: %s. Skipping.", wf_name, exc
                )
                continue

            self.workflows[workflow.workflow_id] = workflow
            await self.engine.submit_workflow(workflow)
            logger.info(
                "Submitted test workflow: %s (%s)", wf_name, wf_config.get("description", "")
            )

    async def start_system(self):
        """Start the entire autonomous agent ecosystem"""
        logger.info("🚀 Starting Autonomous Agent Ecosystem (PRODUCTION MODE)")
        logger.info("=" * 60)

        try:
            # Initialize agents
            await self.initialize_agents()

            # Create sample workflows
            await self.create_sample_workflows()

            # Start engine and monitoring in parallel
            engine_task = asyncio.create_task(self.engine.start_engine())
            monitoring_task = asyncio.create_task(self.dashboard.start_monitoring())

            # Run for specified duration
            duration = self.config.get("duration", 120)
            logger.info(
                f"System running for {duration} seconds... Press Ctrl+C to stop."
            )
            logger.info("=" * 60)

            await asyncio.sleep(duration)

            # Cancel tasks gracefully
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
        except Exception as e:
            logger.error(f"❌ System error: {str(e)}", exc_info=True)
        finally:
            await self.generate_final_report()

    async def generate_final_report(self):
        """Generate a comprehensive final report of system execution"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 FINAL SYSTEM REPORT")
        logger.info("=" * 60)

        # System health summary
        system_health = self.dashboard.get_system_health()
        logger.info(
            f"📈 System Health Score: {system_health['health_score']:.2f} ({system_health['status']})"
        )
        logger.info(f"🤖 Active Agents: {system_health['active_agents_count']}")

        # Workflow execution summary
        logger.info("\n" + "-" * 40)
        logger.info("_WORKFLOW EXECUTION SUMMARY_")
        logger.info("-" * 40)

        for workflow_id, workflow in self.workflows.items():
            completed_tasks = sum(
                1 for t in workflow.tasks if t.task_id in self.engine.completed_tasks
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


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Autonomous Agent Ecosystem")
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Duration to run the system in seconds (default: 120)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )

    # LLM Configuration
    parser.add_argument(
        "--llm-provider",
        type=str,
        help="LLM provider to use (openai, openrouter, ollama)",
    )
    parser.add_argument("--model-name", type=str, help="Model name to use")
    parser.add_argument("--api-key", type=str, help="API key for the LLM provider")
    parser.add_argument("--api-base", type=str, help="Base URL for the LLM API")
    parser.add_argument(
        "--sample-workflows",
        type=str,
        help="Path to sample workflows YAML file (relative to config/ by default)",
    )

    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_arguments()

    # Configure logging
    configure_logging(log_level="DEBUG" if args.debug else "INFO", log_to_file=True)

    # Convert args to config dict
    config = vars(args)
    # Remove None values so they don't override defaults
    config = {k: v for k, v in config.items() if v is not None}

    ecosystem = AutonomousAgentEcosystem(config=config)

    if args.interactive:
        await interactive_mode(ecosystem, args)
    else:
        await ecosystem.start_system()


async def interactive_mode(ecosystem, args):
    """Run interactive mode"""
    try:
        await ecosystem.initialize_agents()

        # Start engine in background
        engine_task = asyncio.create_task(ecosystem.engine.start_engine())
        monitor_task = asyncio.create_task(ecosystem.dashboard.start_monitoring())

        # Setup NL Interface
        llm_config = ecosystem.loaded_configs.get("llm")

        llm_client = LLMFactory.create_client(
            provider=args.llm_provider or getattr(llm_config, "provider", "openai"),
            api_key=args.api_key or getattr(llm_config, "api_key", None),
            base_url=args.api_base or getattr(llm_config, "api_base", None),
        )

        nl_interface = NaturalLanguageInterface(
            engine=ecosystem.engine,
            llm_client=llm_client,
            model_name=args.model_name
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
                # 1. Try to match a template first (Agentic shortcut)
                # In a real system, we'd use the LLM to pick the template.
                # For now, we rely on the NL interface or direct parsing.
                workflow_id = await nl_interface.process_request(user_input)
                print(f"✅ Workflow submitted: {workflow_id}")
            except Exception as e:
                print(f"❌ Error: {e}")

        # Cleanup
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


if __name__ == "__main__":
    asyncio.run(main())
