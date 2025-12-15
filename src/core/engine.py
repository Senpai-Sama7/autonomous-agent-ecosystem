"""
Autonomous Agent Ecosystem Core Engine
November 25, 2025

This engine orchestrates self-coordinating agent networks for complex workflow automation.
Features:
- Incentive alignment mechanisms
- Failure recovery with graceful degradation
- Performance monitoring with emergent behavior detection
- Cost optimization algorithms balancing speed vs. reliability
- Self-healing with circuit breakers
- A2A (Agent-to-Agent) protocol integration
- MCP (Model Context Protocol) integration
- Recursive learning from experiences
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Set, Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from enum import Enum
from .database import DatabaseManager
from .task_queue import TaskQueue, WorkflowPriority

# Structured logging and metrics
from src.utils.structured_logger import get_logger, LogContext, log_performance
from src.monitoring.metrics import get_metrics_collector

# Advanced systems integration
from .self_healing import get_self_healing_system, SelfHealingSystem, HealthStatus
from .circuit_breaker import (
    get_circuit_breaker,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
)
from .a2a_protocol import (
    get_a2a_coordinator,
    A2ACoordinator,
    AgentCard,
    A2ATask,
    A2ATaskState,
)
from .mcp_integration import MCPRegistry, MCPToolExecutor

# Lazy import for recursive learning (has heavy dependencies)
RecursiveLearner = None


def _get_recursive_learner():
    """Lazy import recursive learner to avoid heavy dependency loading at startup."""
    global RecursiveLearner
    if RecursiveLearner is None:
        try:
            from .recursive_learning import get_recursive_learner

            RecursiveLearner = get_recursive_learner
        except ImportError:
            RecursiveLearner = lambda: None
    return RecursiveLearner()


logger = get_logger("AgentEngine")
metrics = get_metrics_collector()


class AgentStatus(Enum):
    """Status of an agent in the ecosystem"""

    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    FAILED = "failed"
    RECOVERING = "recovering"
    DEGRADED = "degraded"


# WorkflowPriority imported from task_queue


@dataclass
class AgentConfig:
    """Configuration for an individual agent"""

    agent_id: str
    capabilities: List[str]
    max_concurrent_tasks: int = 3
    recovery_timeout: float = 30.0  # seconds
    degradation_threshold: float = 0.7  # 70% performance before degradation
    cost_per_operation: float = 1.0
    reliability_score: float = 0.95  # 0.0 to 1.0


@dataclass(order=True)
class Task:
    """Individual task to be executed by agents"""

    task_id: str = field(compare=False)
    description: str = field(compare=False)
    required_capabilities: List[str] = field(default_factory=list, compare=False)
    priority: WorkflowPriority = field(default=WorkflowPriority.MEDIUM, compare=False)
    deadline: Optional[float] = field(default=None, compare=False)
    dependencies: List[str] = field(default_factory=list, compare=False)
    created_at: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)
    retry_count: int = field(default=0, compare=False)  # Track retry attempts
    workflow_id: Optional[str] = field(default=None, compare=False)  # Parent workflow


@dataclass
class Workflow:
    """Complete workflow composed of multiple tasks"""

    workflow_id: str
    name: str
    tasks: List[Task]
    priority: WorkflowPriority = WorkflowPriority.MEDIUM
    max_execution_time: float = 3600.0  # 1 hour default
    cost_budget: float = 1000.0
    success_criteria: Dict[str, Any] = field(default_factory=dict)


class AgentEngine:
    """
    Core engine that manages the autonomous agent ecosystem.

    Thread-safe implementation with proper synchronization for concurrent access.

    Integrated Advanced Systems:
    - Self-Healing: Automatic fault detection and recovery
    - Circuit Breakers: Prevent cascade failures
    - A2A Protocol: Agent-to-agent communication
    - MCP: Model Context Protocol for external tools
    - Recursive Learning: Experience-based pattern matching
    """

    def __init__(
        self, max_metrics_per_agent: int = 1000, enable_advanced_systems: bool = True
    ):
        self.agents: Dict[str, AgentConfig] = {}
        self.agent_instances: Dict[str, Any] = {}  # Holds actual agent instances
        self.workflows: Dict[str, Workflow] = {}
        self._task_queue = TaskQueue()  # Use extracted TaskQueue class
        self.agent_status: Dict[str, AgentStatus] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        self._agent_active_counts: Dict[str, int] = {}
        self.max_metrics_per_agent = max_metrics_per_agent  # Prevent unbounded growth
        self.incentive_pool: float = 10000.0  # Total incentive budget
        self.emergent_behaviors: List[Dict[str, Any]] = []
        self.db = DatabaseManager()
        self._db_initialized = False

        # Shutdown control
        self._shutdown_requested = False
        self._last_status_log = 0.0

        # Synchronization locks for thread-safe operations
        self._agent_lock = asyncio.Lock()
        self._metrics_lock = asyncio.Lock()
        self._workflow_lock = asyncio.Lock()

        # GIL Optimization: Process Pool for CPU-bound tasks (lazy init)
        self._process_pool: Optional[ProcessPoolExecutor] = None

        # Advanced Systems Integration
        self._enable_advanced_systems = enable_advanced_systems
        self._self_healing: Optional[SelfHealingSystem] = None
        self._a2a_coordinator: Optional[A2ACoordinator] = None
        self._mcp_registry: Optional[MCPRegistry] = None
        self._mcp_executor: Optional[MCPToolExecutor] = None
        self._learner = None  # RecursiveLearner (lazy loaded)
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._advanced_systems_started = False

        logger.info(
            "AgentEngine initialized",
            extra={"advanced_systems_enabled": enable_advanced_systems},
        )

    @property
    def process_pool(self) -> ProcessPoolExecutor:
        """Lazy-initialize process pool for CPU-bound tasks."""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=4)
        return self._process_pool

    # Properties for backward compatibility
    @property
    def task_queue(self):
        return self._task_queue._queue

    @property
    def active_tasks(self) -> Dict[str, Task]:
        return self._task_queue.active_tasks

    @property
    def completed_tasks(self) -> Set[str]:
        return self._task_queue.completed_tasks

    @property
    def failed_tasks(self) -> Set[str]:
        return self._task_queue.failed_tasks

    async def register_agent(self, config: AgentConfig, instance: Any = None):
        """Register a new agent with the ecosystem"""
        self.agents[config.agent_id] = config
        if instance:
            self.agent_instances[config.agent_id] = instance

        self.agent_status[config.agent_id] = AgentStatus.IDLE
        self.performance_metrics[config.agent_id] = []
        self._agent_active_counts[config.agent_id] = 0

        # Persist to DB (non-blocking)
        await self.db.save_agent_async(
            config.agent_id,
            asdict(config),
            AgentStatus.IDLE.value,
            config.reliability_score,
        )

        logger.info(
            f"Registered agent {config.agent_id} with capabilities: {config.capabilities}"
        )

    @log_performance
    async def submit_workflow(self, workflow: Workflow):
        """Submit a new workflow for execution"""
        with LogContext(workflow_id=workflow.workflow_id):
            self.workflows[workflow.workflow_id] = workflow
            await self.db.save_workflow_async(
                workflow.workflow_id,
                workflow.name,
                "submitted",
                workflow.priority.value,
            )

            metrics.record_workflow_start()
            logger.info(
                f"Submitted workflow {workflow.workflow_id}",
                extra={
                    "workflow_name": workflow.name,
                    "task_count": len(workflow.tasks),
                    "priority": workflow.priority.value,
                },
            )

            # Break down workflow into individual tasks and queue them
            for task in workflow.tasks:
                task.workflow_id = workflow.workflow_id  # Set parent workflow reference
                await self.db.save_task_async(
                    task.task_id, workflow.workflow_id, task.description, "queued"
                )
                await self._queue_task(task, workflow.priority)

    async def _queue_task(self, task: Task, workflow_priority: WorkflowPriority):
        """Queue a task with appropriate priority"""
        priority_score = TaskQueue.calculate_priority(
            workflow_priority, task.deadline, bool(task.dependencies)
        )
        await self._task_queue.enqueue(task, priority_score)
        logger.debug(f"Queued task {task.task_id} with priority score {priority_score}")

    async def _initialize_advanced_systems(self):
        """Initialize all advanced systems (self-healing, A2A, MCP, learning)."""
        if not self._enable_advanced_systems or self._advanced_systems_started:
            return

        logger.info("Initializing advanced systems...")

        try:
            # Initialize Self-Healing System
            self._self_healing = get_self_healing_system()

            # Register agent health checks with self-healing
            for agent_id, instance in self.agent_instances.items():
                if hasattr(instance, "health_check"):
                    self._self_healing.register_component(
                        component_id=agent_id,
                        health_check=instance.health_check,
                        recovery_callback=instance.recover
                        if hasattr(instance, "recover")
                        else None,
                    )
                # Create circuit breaker for each agent
                self._circuit_breakers[agent_id] = get_circuit_breaker(
                    f"agent_{agent_id}",
                    CircuitBreakerConfig(failure_threshold=3, timeout_seconds=30.0),
                )

            # Initialize A2A Coordinator
            self._a2a_coordinator = get_a2a_coordinator()
            await self._a2a_coordinator.start()

            # Initialize MCP Registry
            self._mcp_registry = MCPRegistry()
            self._mcp_executor = MCPToolExecutor(self._mcp_registry)

            # Initialize Recursive Learner (lazy)
            self._learner = _get_recursive_learner()

            self._advanced_systems_started = True
            logger.info(
                "Advanced systems initialized successfully",
                extra={
                    "self_healing": self._self_healing is not None,
                    "a2a_coordinator": self._a2a_coordinator is not None,
                    "mcp_registry": self._mcp_registry is not None,
                    "learner": self._learner is not None,
                    "circuit_breakers": len(self._circuit_breakers),
                },
            )

        except Exception as e:
            logger.error(f"Failed to initialize advanced systems: {e}", exc_info=True)
            # Continue without advanced systems - graceful degradation
            self._enable_advanced_systems = False

    async def _shutdown_advanced_systems(self):
        """Shutdown all advanced systems gracefully."""
        if not self._advanced_systems_started:
            return

        logger.info("Shutting down advanced systems...")

        try:
            # Stop Self-Healing
            if self._self_healing:
                await self._self_healing.stop()

            # Stop A2A Coordinator
            if self._a2a_coordinator:
                await self._a2a_coordinator.stop()

            self._advanced_systems_started = False
            logger.info("Advanced systems shutdown complete")

        except Exception as e:
            logger.error(f"Error during advanced systems shutdown: {e}")

    async def start_engine(self):
        """Start the agent engine main loop"""
        logger.info("Starting Autonomous Agent Ecosystem Engine")
        self._shutdown_requested = False

        # Initialize database asynchronously
        if not self._db_initialized:
            await self.db.async_init()
            self._db_initialized = True

        # Initialize advanced systems
        await self._initialize_advanced_systems()

        # Start self-healing monitoring if enabled
        if self._self_healing and self._advanced_systems_started:
            await self._self_healing.start()

        engine_task = asyncio.create_task(self._engine_loop())
        monitor_task = asyncio.create_task(self._monitoring_loop())

        # Add learning loop if learner is available
        tasks = [engine_task, monitor_task]
        if self._learner and self._advanced_systems_started:
            learning_task = asyncio.create_task(self._learning_loop())
            tasks.append(learning_task)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Engine tasks cancelled")
        finally:
            logger.info("Engine shutdown complete")

    async def _learning_loop(self):
        """Background loop for periodic batch learning from experiences."""
        while not self._shutdown_requested:
            try:
                if self._learner and hasattr(self._learner, "learn_batch"):
                    result = await self._learner.learn_batch(batch_size=32)
                    if result.get("patterns_added", 0) > 0:
                        logger.debug(f"Learning iteration completed", extra=result)
                await asyncio.sleep(60.0)  # Learn every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(120.0)  # Longer delay after errors

    async def shutdown(self):
        """Request graceful shutdown of the engine"""
        logger.info("Shutdown requested for agent engine")
        self._shutdown_requested = True

        # Shutdown advanced systems
        await self._shutdown_advanced_systems()

        if self.process_pool:
            logger.info("Shutting down process pool...")
            self.process_pool.shutdown(wait=False)

    async def _engine_loop(self):
        """Main engine loop that processes tasks and manages agents"""
        while not self._shutdown_requested:
            try:
                # Get next task from queue
                priority_score, task = await self._task_queue.dequeue()

                # Check dependencies using TaskQueue
                if self._task_queue.has_failed_dependency(task.dependencies):
                    await self._task_queue.mark_failed(task.task_id)
                    continue

                if not self._task_queue.dependencies_met(task.dependencies):
                    await self._task_queue.requeue(task, priority_score + 0.5)
                    await asyncio.sleep(1.0)
                    continue

                # Find suitable agent for this task
                suitable_agent = await self._find_suitable_agent(task)

                if suitable_agent:
                    await self._execute_task_with_agent(task, suitable_agent)
                else:
                    await self._task_queue.requeue(task, priority_score + 0.1)
                    logger.warning(
                        f"No suitable agent found for task {task.task_id}, requeuing"
                    )

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in engine loop: {str(e)}")
                await asyncio.sleep(1.0)  # Recover from errors

    def _is_transient_error(self, error_message: str) -> bool:
        """Determine if an error is transient and worth retrying"""
        transient_keywords = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "rate limit",
            "too many requests",
            "unavailable",
            "service temporarily",
            "try again",
        ]
        error_lower = error_message.lower()
        return any(keyword in error_lower for keyword in transient_keywords)

    async def _find_suitable_agent(self, task: Task) -> Optional[str]:
        """Find the most suitable agent for a given task"""
        suitable_agents = []

        for agent_id, config in self.agents.items():
            agent_status = self.agent_status.get(agent_id, AgentStatus.IDLE)

            # Skip agents that are failed or recovering
            if agent_status in [AgentStatus.FAILED, AgentStatus.RECOVERING]:
                continue

            # Check if agent has required capabilities
            has_capabilities = all(
                cap in config.capabilities for cap in task.required_capabilities
            )

            # Track active assignments per agent
            active_count = self._agent_active_counts.get(agent_id, 0)
            has_capacity = active_count < config.max_concurrent_tasks

            if has_capabilities and has_capacity:
                suitable_agents.append((agent_id, config))

        if not suitable_agents:
            return None

        # Select best agent based on reliability and cost
        best_agent = max(
            suitable_agents,
            key=lambda x: (x[1].reliability_score - (x[1].cost_per_operation * 0.1)),
        )[0]

        return best_agent

    @log_performance
    async def _execute_task_with_agent(self, task: Task, agent_id: str):
        """Execute a task with a specific agent, with circuit breaker protection."""
        start_time = time.time()
        success = False
        result = None

        # Check circuit breaker before executing
        circuit_breaker = self._circuit_breakers.get(agent_id)
        if circuit_breaker:
            try:
                if circuit_breaker.is_open:
                    logger.warning(
                        f"Circuit breaker open for {agent_id}, skipping task {task.task_id}"
                    )
                    # Requeue with delay
                    await asyncio.sleep(2.0)
                    priority_score = TaskQueue.calculate_priority(
                        task.priority, task.deadline, bool(task.dependencies)
                    )
                    await self._task_queue.requeue(task, priority_score + 0.5)
                    return
            except Exception:
                pass  # Continue without circuit breaker

        try:
            self.agent_status[agent_id] = AgentStatus.BUSY
            self._agent_active_counts[agent_id] = (
                self._agent_active_counts.get(agent_id, 0) + 1
            )
            await self._task_queue.mark_active(task.task_id, task)
            workflow_id = task.workflow_id or "standalone"
            await self.db.save_task_async(
                task.task_id,
                workflow_id,
                task.description,
                "running",
                assigned_agent=agent_id,
            )

            metrics.record_task_start(agent_id)
            logger.info(
                f"Executing task {task.task_id}",
                extra={
                    "agent_id": agent_id,
                    "task_type": task.required_capabilities[0]
                    if task.required_capabilities
                    else "generic",
                    "workflow_id": workflow_id,
                },
            )

            # REAL EXECUTION
            agent_instance = self.agent_instances.get(agent_id)
            if not agent_instance:
                raise ValueError(f"Agent instance for {agent_id} not found")

            # Execute using the real agent instance with circuit breaker protection
            from agents.base_agent import AgentContext

            context = AgentContext(
                agent_id=agent_id,
                current_time=time.time(),
                system_load=0.5,  # Placeholder
                available_resources={},
                other_agents=list(self.agents.keys()),
                active_workflows=list(self.workflows.keys()),
                incentive_pool=self.incentive_pool,
            )

            # Execute with optional circuit breaker wrapper
            if circuit_breaker:
                async with circuit_breaker:
                    result = await agent_instance.execute_task(asdict(task), context)
            else:
                result = await agent_instance.execute_task(asdict(task), context)

            execution_time = time.time() - start_time

            # Record performance (non-blocking) with size limit
            self.performance_metrics[agent_id].append(execution_time)
            if len(self.performance_metrics[agent_id]) > self.max_metrics_per_agent:
                # Keep only recent metrics
                self.performance_metrics[agent_id] = self.performance_metrics[agent_id][
                    -self.max_metrics_per_agent :
                ]
            await self.db.log_metric_async(agent_id, "execution_time", execution_time)

            # Process result
            if result.success:
                success = True
                logger.info(
                    f"Task {task.task_id} completed",
                    extra={
                        "agent_id": agent_id,
                        "duration_ms": execution_time * 1000,
                        "status": "success",
                    },
                )
                await self._task_queue.mark_completed(task.task_id)
                workflow_id = task.workflow_id or "standalone"
                await self.db.save_task_async(
                    task.task_id,
                    workflow_id,
                    task.description,
                    "completed",
                    assigned_agent=agent_id,
                    result=result.result_data,
                )

                # Apply incentives based on performance
                await self._apply_incentives(agent_id, execution_time, task.priority)
            else:
                logger.error(
                    f"Task {task.task_id} failed",
                    extra={
                        "agent_id": agent_id,
                        "error": result.error_message,
                        "duration_ms": execution_time * 1000,
                    },
                )

                # Check if task should be retried
                retry_count = getattr(task, "retry_count", 0)
                max_retries = 3

                if retry_count < max_retries and self._is_transient_error(
                    result.error_message or ""
                ):
                    # Exponential backoff: 2^retry_count seconds
                    backoff_time = 2**retry_count
                    logger.info(
                        f"Retrying task {task.task_id}",
                        extra={
                            "attempt": retry_count + 1,
                            "max_retries": max_retries,
                            "backoff_seconds": backoff_time,
                        },
                    )

                    # Update retry count
                    task.retry_count = retry_count + 1

                    # Re-queue task after backoff
                    await asyncio.sleep(backoff_time)
                    priority_score = TaskQueue.calculate_priority(
                        task.priority, task.deadline, bool(task.dependencies)
                    )
                    await self._task_queue.requeue(task, priority_score)
                else:
                    # Max retries exceeded or permanent failure
                    await self._task_queue.mark_failed(task.task_id)
                    workflow_id = task.workflow_id or "standalone"
                    await self.db.save_task_async(
                        task.task_id,
                        workflow_id,
                        task.description,
                        "failed",
                        assigned_agent=agent_id,
                        result={"error": result.error_message},
                    )
                    await self._handle_task_failure(task, agent_id)

        except Exception as e:
            logger.error(
                f"Error executing task {task.task_id}",
                extra={
                    "agent_id": agent_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            metrics.record_error(type(e).__name__)
            await self._handle_task_failure(task, agent_id, str(e))

        finally:
            execution_time = time.time() - start_time
            metrics.record_task_complete(agent_id, execution_time, success)
            self.agent_status[agent_id] = AgentStatus.ACTIVE
            await self._task_queue.remove_active(task.task_id)
            current_count = self._agent_active_counts.get(agent_id, 0)
            if current_count <= 1:
                self._agent_active_counts[agent_id] = 0
            else:
                self._agent_active_counts[agent_id] = current_count - 1

            # Record learning experience for pattern matching
            if self._learner and self._advanced_systems_started:
                try:
                    from .recursive_learning import ExperienceType

                    context = {
                        "agent_id": agent_id,
                        "task_type": task.required_capabilities[0]
                        if task.required_capabilities
                        else "generic",
                        "priority": str(task.priority),
                        "has_dependencies": bool(task.dependencies),
                    }
                    outcome = {
                        "execution_time": execution_time,
                        "success": success,
                        "result_summary": str(result.result_data)[:100]
                        if result and result.success
                        else None,
                    }
                    # Reward: 1.0 for fast success, 0.5 for slow success, -0.5 for failure
                    if success:
                        reward = 1.0 if execution_time < 5.0 else 0.5
                    else:
                        reward = -0.5

                    self._learner.record_experience(
                        experience_type=ExperienceType.TASK_COMPLETION,
                        context=context,
                        action=f"execute_{task.required_capabilities[0] if task.required_capabilities else 'generic'}",
                        outcome=outcome,
                        reward=reward,
                    )
                except Exception as e:
                    logger.debug(f"Failed to record learning experience: {e}")

    async def _apply_incentives(
        self, agent_id: str, execution_time: float, priority: WorkflowPriority
    ):
        """Apply incentive rewards based on performance"""
        config = self.agents[agent_id]

        # Base reward based on priority
        priority_reward = {
            WorkflowPriority.CRITICAL: 100.0,
            WorkflowPriority.HIGH: 50.0,
            WorkflowPriority.MEDIUM: 25.0,
            WorkflowPriority.LOW: 10.0,
        }[priority]

        # Performance bonus (faster execution = higher bonus)
        performance_bonus = max(
            0, (5.0 - execution_time) * 10
        )  # Max 5 seconds baseline

        # Total reward
        total_reward = priority_reward + performance_bonus

        # Update agent's reliability score based on performance
        if execution_time < 2.0:  # Good performance threshold
            config.reliability_score = min(0.99, config.reliability_score + 0.01)
        else:
            config.reliability_score = max(0.5, config.reliability_score - 0.02)

        logger.info(
            f"Agent {agent_id} earned reward of ${total_reward:.2f} for fast execution"
        )

    async def _handle_task_failure(self, task: Task, agent_id: str, error: str = ""):
        """Handle task failure with recovery strategies"""
        config = self.agents[agent_id]

        # Update agent status to failed
        self.agent_status[agent_id] = AgentStatus.FAILED
        logger.warning(
            f"Agent {agent_id} failed task {task.task_id}. Initiating recovery..."
        )

        # Attempt recovery
        recovery_success = await self._attempt_agent_recovery(agent_id)

        if recovery_success:
            logger.info(f"Agent {agent_id} recovered successfully")
            # Requeue the failed task with high priority
            priority_score = TaskQueue.calculate_priority(
                WorkflowPriority.HIGH, task.deadline, bool(task.dependencies)
            )
            await self._task_queue.enqueue(task, priority_score)
        else:
            logger.error(
                f"Agent {agent_id} recovery failed. Task {task.task_id} needs manual intervention"
            )
            # Implement fallback strategies
            await self._trigger_fallback_strategy(task)

    async def _attempt_agent_recovery(self, agent_id: str) -> bool:
        """Attempt to recover a failed agent"""
        config = self.agents[agent_id]
        start_time = time.time()

        logger.info(f"Starting recovery for agent {agent_id}")
        self.agent_status[agent_id] = AgentStatus.RECOVERING

        # Real recovery logic: Check if agent instance is responsive
        # In this architecture, we just reset the state and hope the next task works
        # A real system might restart a container here.
        await asyncio.sleep(1.0)

        self.agent_status[agent_id] = AgentStatus.ACTIVE
        logger.info(f"Agent {agent_id} recovered successfully (state reset)")
        return True

    async def _trigger_fallback_strategy(self, task: Task):
        """Trigger fallback strategies when primary agent recovery fails"""
        logger.warning(f"Triggering fallback strategies for task {task.task_id}")

        # Strategy 1: Try alternative agents with same capabilities
        alternative_agents = [
            agent_id
            for agent_id, config in self.agents.items()
            if all(cap in config.capabilities for cap in task.required_capabilities)
            and self.agent_status[agent_id] == AgentStatus.ACTIVE
        ]

        if alternative_agents:
            # Select best alternative agent
            best_alternative = max(
                alternative_agents, key=lambda aid: self.agents[aid].reliability_score
            )
            logger.info(
                f"Found alternative agent {best_alternative} for task {task.task_id}"
            )
            await self._execute_task_with_agent(task, best_alternative)
            return

        # Strategy 2: Degraded mode - use agents with partial capabilities
        partial_agents = [
            (
                agent_id,
                sum(
                    1
                    for cap in task.required_capabilities
                    if cap in self.agents[agent_id].capabilities
                ),
            )
            for agent_id, config in self.agents.items()
            if self.agent_status[agent_id] == AgentStatus.ACTIVE
        ]

        partial_agents.sort(key=lambda x: x[1], reverse=True)

        if partial_agents and partial_agents[0][1] > 0:
            # Use agent with most matching capabilities
            degraded_agent = partial_agents[0][0]
            logger.warning(
                f"Using degraded mode with agent {degraded_agent} for task {task.task_id}"
            )
            self.agent_status[degraded_agent] = AgentStatus.DEGRADED
            await self._execute_task_with_agent(task, degraded_agent)
            return

        # Strategy 3: Manual intervention required
        logger.error(
            f"No fallback agents available for task {task.task_id}. Manual intervention required."
        )

    async def _monitoring_loop(self):
        """Continuous monitoring loop for system health and emergent behaviors"""
        while not self._shutdown_requested:
            try:
                await self._check_system_health()
                await self._detect_emergent_behaviors()
                await self._optimize_resource_allocation()

                # Log system status periodically (every 60 seconds)
                current_time = time.time()
                if current_time - self._last_status_log >= 60.0:
                    self._log_system_status()
                    self._last_status_log = current_time

                await asyncio.sleep(5.0)  # Monitor every 5 seconds

            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(10.0)  # Longer delay after errors

    async def _check_system_health(self):
        """Check overall system health and agent status"""
        healthy_agents = sum(
            1
            for status in self.agent_status.values()
            if status in [AgentStatus.ACTIVE, AgentStatus.IDLE]
        )
        total_agents = len(self.agent_status)

        health_ratio = healthy_agents / total_agents if total_agents > 0 else 0

        if health_ratio < 0.5:
            logger.warning(
                f"System health critical: {healthy_agents}/{total_agents} agents healthy"
            )
        elif health_ratio < 0.8:
            logger.warning(
                f"System health degraded: {healthy_agents}/{total_agents} agents healthy"
            )

    async def _detect_emergent_behaviors(self):
        """Detect unexpected patterns or emergent behaviors in agent interactions"""
        # This is a simplified version - in real system, this would use ML to detect patterns
        current_metrics = {
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize(),
            "agent_distribution": {
                str(status): sum(1 for s in self.agent_status.values() if s == status)
                for status in AgentStatus
            },
        }

        # Check for unusual patterns
        if current_metrics["queue_size"] > 100 and current_metrics["active_tasks"] < 5:
            self.emergent_behaviors.append(
                {
                    "timestamp": time.time(),
                    "behavior": "task_queue_backlog",
                    "description": "Tasks are queuing up but not being processed",
                    "severity": "high",
                }
            )
            logger.warning("Detected task queue backlog behavior")

        # Log significant emergent behaviors
        if len(self.emergent_behaviors) > 10:
            logger.info(f"Emergent behavior detected: {self.emergent_behaviors[-1]}")

    async def _optimize_resource_allocation(self):
        """Optimize resource allocation based on current workload and agent performance"""
        # This is a simplified optimization routine
        for agent_id, metrics in self.performance_metrics.items():
            if len(metrics) > 10:  # Enough data for analysis
                avg_performance = sum(metrics[-10:]) / 10

                # Adjust agent configuration based on performance
                config = self.agents[agent_id]
                if avg_performance < 1.0:  # Very fast performance
                    # Increase capacity
                    config.max_concurrent_tasks = min(
                        10, config.max_concurrent_tasks + 1
                    )
                elif avg_performance > 10.0:  # Slow performance
                    # Decrease capacity or trigger investigation
                    config.max_concurrent_tasks = max(
                        1, config.max_concurrent_tasks - 1
                    )

    def _log_system_status(self):
        """Log current system status for monitoring"""
        status_summary = {
            "timestamp": time.time(),
            "total_agents": len(self.agents),
            "agent_status_counts": {
                str(status): sum(1 for s in self.agent_status.values() if s == status)
                for status in AgentStatus
            },
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize(),
            "incentive_pool": self.incentive_pool,
        }

        logger.info(f"System Status: {json.dumps(status_summary, indent=2)}")

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics for external monitoring"""
        return {
            "agent_count": len(self.agents),
            "active_workflows": len(self.workflows),
            "pending_tasks": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "agent_status_distribution": {
                str(status): sum(1 for s in self.agent_status.values() if s == status)
                for status in AgentStatus
            },
            "average_performance": {
                agent_id: sum(metrics[-5:]) / 5 if len(metrics) >= 5 else None
                for agent_id, metrics in self.performance_metrics.items()
            },
        }

    async def execute_cpu_bound(self, func: Callable, *args) -> Any:
        """
        Execute a CPU-bound function in a separate process to bypass GIL.
        Useful for AST parsing, heavy computation, or complex text synthesis.
        """
        if not self._process_pool:
            self._process_pool = ProcessPoolExecutor()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._process_pool, func, *args)
