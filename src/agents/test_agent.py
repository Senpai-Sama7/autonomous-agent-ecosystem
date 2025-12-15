"""Test execution agent for running test suites."""

import subprocess
import logging
from typing import Dict, Any
from .base_agent import BaseAgent, AgentCapability, AgentContext, TaskResult

logger = logging.getLogger(__name__)


class TestAgent(BaseAgent):
    """Executes test suites and reports results."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, [AgentCapability.DATA_PROCESSING], config)

    async def execute_task(
        self, task: Dict[str, Any], context: AgentContext
    ) -> TaskResult:
        """Run tests."""
        try:
            command = task.get("command")
            target_file = task.get("target_file")

            if not command:
                return TaskResult(
                    success=False, error_message="No test command provided"
                )

            cmd = command.split() if isinstance(command, str) else command
            if target_file:
                cmd.append(target_file)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            return TaskResult(
                success=result.returncode == 0,
                result_data={
                    "output": result.stdout,
                    "errors": result.stderr,
                    "return_code": result.returncode,
                },
            )
        except subprocess.TimeoutExpired:
            return TaskResult(success=False, error_message="Test execution timed out")
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return TaskResult(success=False, error_message=str(e))
