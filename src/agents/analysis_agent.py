"""Code analysis agent for static analysis and linting."""

import asyncio
import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentCapability, AgentContext, TaskResult

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAgent):
    """Runs static analysis and linting tools."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, [AgentCapability.DATA_PROCESSING], config)
        self.command_timeout = config.get("command_timeout", 60)

    def _build_command(self, tool: str, path: str, arguments: Any) -> List[str]:
        if not tool or not path:
            raise ValueError("Tool and path required")

        cmd = [tool, path]
        if arguments:
            if isinstance(arguments, list):
                cmd.extend(arguments)
            else:
                cmd.append(str(arguments))
        return cmd

    async def _run_analysis(self, cmd: List[str], tool: str, path: str) -> TaskResult:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.command_timeout
            )
            return TaskResult(
                success=proc.returncode == 0,
                result_data={
                    "output": stdout.decode("utf-8", errors="replace"),
                    "warnings": stderr.decode("utf-8", errors="replace"),
                    "tool": tool,
                    "path": path,
                    "return_code": proc.returncode,
                },
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return TaskResult(
                success=False,
                error_message=f"Analysis timed out after {self.command_timeout}s",
            )
        except FileNotFoundError:
            return TaskResult(success=False, error_message=f"Tool not found: {tool}")

    async def execute_task(self, task: Dict[str, Any], context: AgentContext) -> TaskResult:
        """Run code analysis."""
        try:
            tool = task.get("tool")
            path = task.get("path")
            arguments = task.get("arguments")
            cmd = self._build_command(tool, path, arguments)
            return await self._run_analysis(cmd, tool, path)
        except ValueError as e:
            return TaskResult(success=False, error_message=str(e))
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return TaskResult(success=False, error_message=str(e))
