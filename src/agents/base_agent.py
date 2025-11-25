"""BaseAgent and supporting classes for the Autonomous Agent Ecosystem"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Enumerates generic capabilities that agents may possess."""
    DATA_PROCESSING = "data_processing"
    API_INTEGRATION = "api_integration"
    DECISION_MAKING = "decision_making"
    OPTIMIZATION = "optimization"
    # Extend as needed for specific agents


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


@dataclass
class TaskResult:
    """Result container returned by agents after executing a task."""
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None

    def __post_init__(self):
        if self.success and not self.result_data:
            self.result_data = {}
        if not self.success and not self.error_message:
            self.error_message = "Unknown error"


class BaseAgent:
    """Base class providing common functionality for all agents."""
    def __init__(self, agent_id: str, capabilities: List[AgentCapability], config: Dict[str, Any]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.config = config
        self.state = AgentState.IDLE
        self.max_task_history = config.get('max_task_history', 1000)  # Prevent memory leaks
        self.task_history: List[TaskResult] = []
        logger.info(f"BaseAgent {agent_id} initialized with capabilities: {[c.value for c in capabilities]}")

    def record_task_result(self, result: TaskResult) -> None:
        """Store the result of a completed task for later analysis."""
        self.task_history.append(result)
        
        # Trim history if it exceeds max size
        if len(self.task_history) > self.max_task_history:
            self.task_history = self.task_history[-self.max_task_history:]
            
        logger.debug(f"Agent {self.agent_id} recorded task result: {result}")

    async def recover(self) -> bool:
        """Placeholder recovery logic – can be overridden by subclasses."""
        logger.info(f"Attempting recovery for agent {self.agent_id}")
        await asyncio.sleep(1)  # Simulate recovery delay
        # Simple heuristic: if there are recent failures, pretend recovery succeeded
        recent_failures = [t for t in self.task_history[-5:] if not t.success]
        if recent_failures:
            self.state = AgentState.ACTIVE
            logger.info(f"Agent {self.agent_id} recovered successfully")
            return True
        logger.warning(f"Agent {self.agent_id} recovery not needed or failed")
        return False

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
            "task_count": len(self.task_history)
        }

    # Subclasses must implement execute_task and optionally self_diagnose
    async def execute_task(self, task: Dict[str, Any], context: AgentContext) -> TaskResult:
        raise NotImplementedError("Subclasses must implement execute_task")

    async def self_diagnose(self) -> Dict[str, Any]:
        """Return basic diagnostic information about the agent."""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "tasks_executed": len(self.task_history),
            "success_rate": self.get_success_rate(),
            "capabilities": [c.value for c in self.capabilities],
        }
