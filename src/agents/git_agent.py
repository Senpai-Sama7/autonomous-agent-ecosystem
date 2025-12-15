"""Git operations agent for version control management."""

import asyncio
import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentCapability, AgentContext, TaskResult

logger = logging.getLogger(__name__)


class GitAgent(BaseAgent):
    """Manages git operations: status, diff, add, commit, checkout, branch, log."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, [AgentCapability.DATA_PROCESSING], config)
        self.command_timeout = config.get("command_timeout", 30)

    async def _run_command(self, cmd: List[str], operation: str) -> TaskResult:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.command_timeout
            )
            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")
            if proc.returncode != 0:
                return TaskResult(success=False, error_message=stderr_text)
            return TaskResult(
                success=True,
                result_data={"output": stdout_text, "operation": operation},
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return TaskResult(
                success=False,
                error_message=f"Git command timed out after {self.command_timeout}s",
            )
        except FileNotFoundError:
            return TaskResult(success=False, error_message="Git binary not found.")

    async def execute_task(self, task: Dict[str, Any], context: AgentContext) -> TaskResult:
        """Execute git operations."""
        try:
            operation = task.get("operation")
            arguments = task.get("arguments", [])

            if operation not in ["status", "diff", "add", "commit", "checkout", "branch", "log"]:
                return TaskResult(success=False, error_message=f"Unknown operation: {operation}")

            cmd = ["git", operation] + arguments
            return await self._run_command(cmd, operation)
        except Exception as e:
            logger.error(f"Git operation failed: {e}")
            return TaskResult(success=False, error_message=str(e))
