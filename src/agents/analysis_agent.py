"""Code analysis agent for static analysis and linting."""

import subprocess
import logging
from typing import Dict, Any
from .base_agent import BaseAgent, AgentCapability, AgentContext, TaskResult

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAgent):
    """Runs static analysis and linting tools."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, [AgentCapability.DATA_PROCESSING], config)

    async def execute_task(
        self, task: Dict[str, Any], context: AgentContext
    ) -> TaskResult:
        """Run code analysis."""
        try:
            tool = task.get("tool")
            path = task.get("path")

            if not tool or not path:
                return TaskResult(success=False, error_message="Tool and path required")

            cmd = [tool, path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            return TaskResult(
                success=result.returncode == 0,
                result_data={
                    "output": result.stdout,
                    "warnings": result.stderr,
                    "tool": tool,
                    "path": path,
                },
            )
        except subprocess.TimeoutExpired:
            return TaskResult(success=False, error_message="Analysis timed out")
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return TaskResult(success=False, error_message=str(e))
