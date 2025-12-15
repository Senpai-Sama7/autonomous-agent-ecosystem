"""Test execution agent for running test suites."""

import asyncio
import logging
import shlex
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentCapability, AgentContext, TaskResult

logger = logging.getLogger(__name__)


class TestAgent(BaseAgent):
    __test__ = False  # Prevent pytest from collecting this class as a test
    """Executes test suites and reports results."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, [AgentCapability.DATA_PROCESSING], config)
        self.command_timeout = config.get("command_timeout", 120)

    def _build_command(self, command: Any, target_file: str | None) -> List[str]:
        if not command:
            raise ValueError("No test command provided")

        if isinstance(command, str):
            cmd = shlex.split(command)
        else:
            cmd = list(command)

        if target_file:
            cmd.append(target_file)
        return cmd

    async def _run_tests(self, cmd: List[str]) -> TaskResult:
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
            return TaskResult(
                success=proc.returncode == 0,
                result_data={
                    "output": stdout_text,
                    "errors": stderr_text,
                    "return_code": proc.returncode,
                },
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return TaskResult(
                success=False,
                error_message=f"Test execution timed out after {self.command_timeout}s",
            )
        except FileNotFoundError:
            return TaskResult(
                success=False, error_message=f"Executable not found: {cmd[0]}"
            )

    async def execute_task(self, task: Dict[str, Any], context: AgentContext) -> TaskResult:
        """Run tests."""
        try:
            command = task.get("command")
            target_file = task.get("target_file")
            cmd = self._build_command(command, target_file)
            return await self._run_tests(cmd)
        except ValueError as e:
            return TaskResult(success=False, error_message=str(e))
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return TaskResult(success=False, error_message=str(e))
