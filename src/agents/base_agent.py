"""BaseAgent and supporting classes for the Autonomous Agent Ecosystem"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from functools import wraps

# Structured logging and metrics
from src.utils.structured_logger import get_logger, log_performance
from src.monitoring.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class AgentCapability(Enum):
    """Enumerates generic capabilities that agents may possess."""

    DATA_PROCESSING = "data_processing"
    API_INTEGRATION = "api_integration"
    DECISION_MAKING = "decision_making"
    OPTIMIZATION = "optimization"
    VERSION_CONTROL = "version_control"
    CODE_ANALYSIS = "code_analysis"
    FILE_OPERATIONS = "file_operations"
    WEB_SEARCH = "web_search"
    MEMORY = "memory"
    TESTING = "testing"


class AgentState(Enum):
    """Simple lifecycle states for an agent."""

    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    FAILED = "failed"
    RECOVERING = "recovering"
    DEGRADED = "degraded"


@dataclass
class AgentContext:
    """Context information passed to agents during task execution."""

    agent_id: str
    current_time: float
    system_load: float
    available_resources: Dict[str, Any]
    other_agents: List[str]
    active_workflows: List[str]
    incentive_pool: float
    timeout: float = 60.0  # Default timeout in seconds
    request_id: Optional[str] = None  # For tracing


@dataclass
class TaskResult:
    """Result container returned by agents after executing a task."""

    success: bool
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    retryable: bool = False  # Indicates if failure is transient

    def __post_init__(self):
        if self.success and not self.result_data:
            self.result_data = {}
        if not self.success and not self.error_message:
            self.error_message = "Unknown error"


class BaseAgent:
    """
    Base class providing common functionality for all agents.

    Features:
    - Timeout handling for all task executions
    - Automatic state management
    - Task history with bounded memory
    - Health check and diagnostics
    - Recovery mechanisms
    """

    DEFAULT_TIMEOUT = 60.0  # seconds
    MAX_RETRIES = 3

    def __init__(
        self, agent_id: str, capabilities: List[AgentCapability], config: Dict[str, Any]
    ):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.config = config
        self.state = AgentState.IDLE
        self.max_task_history = config.get("max_task_history", 1000)
        self.default_timeout = config.get("timeout", self.DEFAULT_TIMEOUT)
        self.task_history: List[TaskResult] = []
        self._consecutive_failures = 0
        self._lock = asyncio.Lock()

        # Initialize metrics
        metrics.update_agent_health(agent_id, True)
        metrics.update_agent_reliability(
            agent_id, config.get("reliability_score", 0.95)
        )

        logger.info(
            f"BaseAgent {agent_id} initialized",
            extra={
                "capabilities": [c.value for c in capabilities],
                "timeout": self.default_timeout,
            },
        )

    @log_performance
    async def execute_with_timeout(
        self, task: Dict[str, Any], context: AgentContext
    ) -> TaskResult:
        """
        Execute task with timeout, state management, and error handling.
        This is the main entry point that wraps execute_task.
        """
        timeout = context.timeout if context.timeout else self.default_timeout
        start_time = time.time()
        task_id = task.get("task_id", "unknown")

        async with self._lock:
            self.state = AgentState.BUSY

        try:
            result = await asyncio.wait_for(
                self.execute_task(task, context), timeout=timeout
            )
            result.execution_time = time.time() - start_time

            if result.success:
                self._consecutive_failures = 0
                logger.info(
                    f"Task completed",
                    extra={
                        "agent_id": self.agent_id,
                        "task_id": task_id,
                        "duration_ms": result.execution_time * 1000,
                    },
                )
            else:
                self._consecutive_failures += 1
                logger.warning(
                    f"Task failed",
                    extra={
                        "agent_id": self.agent_id,
                        "task_id": task_id,
                        "error": result.error_message,
                    },
                )

            self.record_task_result(result)
            return result

        except asyncio.TimeoutError:
            self._consecutive_failures += 1
            result = TaskResult(
                success=False,
                error_message=f"Task timed out after {timeout}s",
                execution_time=time.time() - start_time,
                retryable=True,
            )
            self.record_task_result(result)
            logger.error(
                f"Task timed out",
                extra={
                    "agent_id": self.agent_id,
                    "task_id": task_id,
                    "timeout": timeout,
                },
            )
            metrics.record_error("TimeoutError")
            return result

        except Exception as e:
            self._consecutive_failures += 1
            result = TaskResult(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                retryable=self._is_retryable_error(e),
            )
            self.record_task_result(result)
            logger.error(
                f"Task exception",
                extra={
                    "agent_id": self.agent_id,
                    "task_id": task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            metrics.record_error(type(e).__name__)
            return result

        finally:
            async with self._lock:
                if self._consecutive_failures >= 3:
                    self.state = AgentState.DEGRADED
                    metrics.update_agent_health(self.agent_id, False)
                elif self._consecutive_failures >= 5:
                    self.state = AgentState.FAILED
                else:
                    self.state = AgentState.IDLE

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is transient and worth retrying."""
        retryable_patterns = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "rate limit",
            "too many requests",
            "unavailable",
            "service temporarily",
            "try again",
            "EAGAIN",
        ]
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in retryable_patterns)

    def record_task_result(self, result: TaskResult) -> None:
        """Store the result of a completed task for later analysis."""
        self.task_history.append(result)
        if len(self.task_history) > self.max_task_history:
            self.task_history = self.task_history[-self.max_task_history :]
        logger.debug(
            f"Agent {self.agent_id} recorded task result: success={result.success}"
        )

    async def recover(self) -> bool:
        """Attempt to recover from failed state."""
        logger.info(f"Attempting recovery for agent {self.agent_id}")
        async with self._lock:
            self.state = AgentState.RECOVERING

        await asyncio.sleep(1)  # Brief pause before recovery

        async with self._lock:
            self._consecutive_failures = 0
            self.state = AgentState.IDLE

        logger.info(f"Agent {self.agent_id} recovered successfully")
        return True

    def get_success_rate(self) -> float:
        """Calculate the success rate from task history."""
        if not self.task_history:
            return 1.0
        successful = sum(1 for t in self.task_history if t.success)
        return successful / len(self.task_history)

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the agent."""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "healthy": self.state not in [AgentState.FAILED, AgentState.DEGRADED],
            "success_rate": self.get_success_rate(),
            "task_count": len(self.task_history),
            "consecutive_failures": self._consecutive_failures,
        }

    async def execute_task(
        self, task: Dict[str, Any], context: AgentContext
    ) -> TaskResult:
        """Subclasses must implement this method."""
        raise NotImplementedError("Subclasses must implement execute_task")

    async def self_diagnose(self) -> Dict[str, Any]:
        """Return diagnostic information about the agent."""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "tasks_executed": len(self.task_history),
            "success_rate": self.get_success_rate(),
            "capabilities": [c.value for c in self.capabilities],
            "consecutive_failures": self._consecutive_failures,
            "config": {k: v for k, v in self.config.items() if not k.startswith("_")},
        }
