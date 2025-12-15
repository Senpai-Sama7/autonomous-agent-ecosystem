"""Git operations agent for version control management."""

import subprocess
import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentCapability, AgentContext, TaskResult

logger = logging.getLogger(__name__)


class GitAgent(BaseAgent):
    """Manages git operations: status, diff, add, commit, checkout, branch, log."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, [AgentCapability.DATA_PROCESSING], config)

    async def execute_task(
        self, task: Dict[str, Any], context: AgentContext
    ) -> TaskResult:
        """Execute git operations."""
        try:
            operation = task.get("operation")
            arguments = task.get("arguments", [])

            if operation not in [
                "status",
                "diff",
                "add",
                "commit",
                "checkout",
                "branch",
                "log",
            ]:
                return TaskResult(
                    success=False, error_message=f"Unknown operation: {operation}"
                )

            cmd = ["git", operation] + arguments
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                return TaskResult(success=False, error_message=result.stderr)

            return TaskResult(
                success=True,
                result_data={"output": result.stdout, "operation": operation},
            )
        except Exception as e:
            logger.error(f"Git operation failed: {e}")
            return TaskResult(success=False, error_message=str(e))
