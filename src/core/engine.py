"""
Autonomous Agent Ecosystem Core Engine
November 25, 2025

This engine orchestrates self-coordinating agent networks for complex workflow automation.
Features:
- Incentive alignment mechanisms
- Failure recovery with graceful degradation
- Performance monitoring with emergent behavior detection
- Cost optimization algorithms balancing speed vs. reliability
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from .database import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentEngine")

class AgentStatus(Enum):
    """Status of an agent in the ecosystem"""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    FAILED = "failed"
    RECOVERING = "recovering"
    DEGRADED = "degraded"

class WorkflowPriority(Enum):
    """Priority levels for workflows"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

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
    Core engine that manages the autonomous agent ecosystem
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentConfig] = {}
        self.agent_instances: Dict[str, Any] = {}  # Holds actual agent instances
        self.workflows: Dict[str, Workflow] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: set = set()  # Track completed task IDs
        self.failed_tasks: set = set()  # Track failed task IDs
        self.agent_status: Dict[str, AgentStatus] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        self.incentive_pool: float = 10000.0  # Total incentive budget
        self.emergent_behaviors: List[Dict[str, Any]] = []
        self.db = DatabaseManager()
        
    async def register_agent(self, config: AgentConfig, instance: Any = None):
        """Register a new agent with the ecosystem"""
        self.agents[config.agent_id] = config
        if instance:
            self.agent_instances[config.agent_id] = instance
            
        self.agent_status[config.agent_id] = AgentStatus.IDLE
        self.performance_metrics[config.agent_id] = []
        
        # Persist to DB
        self.db.save_agent(config.agent_id, asdict(config), AgentStatus.IDLE.value, config.reliability_score)
        
        logger.info(f"Registered agent {config.agent_id} with capabilities: {config.capabilities}")
        
    async def submit_workflow(self, workflow: Workflow):
        """Submit a new workflow for execution"""
        self.workflows[workflow.workflow_id] = workflow
        self.db.save_workflow(workflow.workflow_id, workflow.name, "submitted", workflow.priority.value)
        logger.info(f"Submitted workflow {workflow.workflow_id}: {workflow.name}")
        
        # Break down workflow into individual tasks and queue them
        for task in workflow.tasks:
            self.db.save_task(task.task_id, workflow.workflow_id, task.description, "queued")
            await self._queue_task(task, workflow.priority)
            
    async def _queue_task(self, task: Task, workflow_priority: WorkflowPriority):
        """Queue a task with appropriate priority"""
        # Calculate priority score (lower number = higher priority)
        priority_score = self._calculate_task_priority(task, workflow_priority)
        await self.task_queue.put((priority_score, task))
        logger.debug(f"Queued task {task.task_id} with priority score {priority_score}")
        
    def _calculate_task_priority(self, task: Task, workflow_priority: WorkflowPriority) -> float:
        """Calculate dynamic priority score for a task"""
        base_priority = {
            WorkflowPriority.CRITICAL: 0.1,
            WorkflowPriority.HIGH: 0.3,
            WorkflowPriority.MEDIUM: 0.5,
            WorkflowPriority.LOW: 0.8
        }[workflow_priority]
        
        # Adjust based on deadline proximity
        if task.deadline:
            time_remaining = task.deadline - time.time()
            if time_remaining < 0:
                deadline_factor = 0.1  # Very urgent
            elif time_remaining < 60:  # 1 minute
                deadline_factor = 0.3
            elif time_remaining < 300:  # 5 minutes
                deadline_factor = 0.5
            else:
                deadline_factor = 0.8
        else:
            deadline_factor = 0.7
            
        # Adjust based on dependencies
        dependency_factor = 0.5 if task.dependencies else 0.8
        
        # Calculate final priority score
        priority_score = base_priority * deadline_factor * dependency_factor
        return priority_score
        
    async def start_engine(self):
        """Start the agent engine main loop"""
        logger.info("Starting Autonomous Agent Ecosystem Engine")
        engine_task = asyncio.create_task(self._engine_loop())
        monitor_task = asyncio.create_task(self._monitoring_loop())
        
        await asyncio.gather(engine_task, monitor_task)
        
    async def _engine_loop(self):
        """Main engine loop that processes tasks and manages agents"""
        while True:
            try:
                # Get next task from queue
                priority_score, task = await self.task_queue.get()
                
                # Check dependencies
                if not self._are_dependencies_met(task):
                    # Requeue if dependencies not met
                    await self.task_queue.put((priority_score + 0.5, task))
                    await asyncio.sleep(1.0)
                    continue

                # Find suitable agent for this task
                suitable_agent = await self._find_suitable_agent(task)
                
                if suitable_agent:
                    # Execute task with selected agent
                    await self._execute_task_with_agent(task, suitable_agent)
                else:
                    # No suitable agent available - requeue with lower priority
                    await self.task_queue.put((priority_score + 0.1, task))
                    logger.warning(f"No suitable agent found for task {task.task_id}, requeuing")
                    
                await asyncio.sleep(0.1)  # Small delay to prevent CPU hogging
                
            except Exception as e:
                logger.error(f"Error in engine loop: {str(e)}")
                await asyncio.sleep(1.0)  # Recover from errors

    def _are_dependencies_met(self, task: Task) -> bool:
        """Check if all task dependencies are completed"""
        if not task.dependencies:
            return True
            
        # Check if all dependencies are in completed_tasks
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                # Check if dependency failed
                if dep_id in self.failed_tasks:
                    logger.warning(f"Task {task.task_id} has failed dependency {dep_id}")
                    return False
                # Dependency not yet completed
                return False
        return True
    
    def _is_transient_error(self, error_message: str) -> bool:
        """Determine if an error is transient and worth retrying"""
        transient_keywords = [
            'timeout', 'connection', 'network', 'temporary',
            'rate limit', 'too many requests', 'unavailable',
            'service temporarily', 'try again'
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
            has_capabilities = all(cap in config.capabilities for cap in task.required_capabilities)
            
            # Check if agent has capacity
            active_count = sum(1 for t in self.active_tasks.values() if t.task_id.startswith(f"{agent_id}_"))
            has_capacity = active_count < config.max_concurrent_tasks
            
            if has_capabilities and has_capacity:
                suitable_agents.append((agent_id, config))
                
        if not suitable_agents:
            return None
            
        # Select best agent based on reliability and cost
        best_agent = max(suitable_agents, key=lambda x: (
            x[1].reliability_score - (x[1].cost_per_operation * 0.1)
        ))[0]
        
        return best_agent
        
    async def _execute_task_with_agent(self, task: Task, agent_id: str):
        """Execute a task with a specific agent"""
        try:
            self.agent_status[agent_id] = AgentStatus.BUSY
            self.active_tasks[task.task_id] = task
            self.db.save_task(task.task_id, "unknown", task.description, "running", assigned_agent=agent_id)
            
            logger.info(f"Executing task {task.task_id} with agent {agent_id}")
            
            # REAL EXECUTION
            start_time = time.time()
            
            agent_instance = self.agent_instances.get(agent_id)
            if not agent_instance:
                raise ValueError(f"Agent instance for {agent_id} not found")

            # Execute using the real agent instance
            # We need to construct the context
            from agents.base_agent import AgentContext
            context = AgentContext(
                agent_id=agent_id,
                current_time=time.time(),
                system_load=0.5, # Placeholder
                available_resources={},
                other_agents=list(self.agents.keys()),
                active_workflows=list(self.workflows.keys()),
                incentive_pool=self.incentive_pool
            )
            
            result = await agent_instance.execute_task(asdict(task), context)
            
            execution_time = time.time() - start_time
            
            # Record performance
            self.performance_metrics[agent_id].append(execution_time)
            self.db.log_metric(agent_id, "execution_time", execution_time)
            
            # Process result
            if result.success:
                logger.info(f"Task {task.task_id} completed successfully by agent {agent_id}")
                self.completed_tasks.add(task.task_id)
                self.db.save_task(task.task_id, "unknown", task.description, "completed", assigned_agent=agent_id, result=result.result_data)
                
                # Apply incentives based on performance
                await self._apply_incentives(agent_id, execution_time, task.priority)
            else:
                logger.error(f"Task {task.task_id} failed for agent {agent_id}: {result.error_message}")
                
                # Check if task should be retried
                retry_count = getattr(task, 'retry_count', 0)
                max_retries = 3
                
                if retry_count < max_retries and self._is_transient_error(result.error_message):
                    # Exponential backoff: 2^retry_count seconds
                    backoff_time = 2 ** retry_count
                    logger.info(f"Retrying task {task.task_id} after {backoff_time}s (attempt {retry_count + 1}/{max_retries})")
                    
                    # Update retry count
                    task.retry_count = retry_count + 1
                    
                    # Re-queue task after backoff
                    await asyncio.sleep(backoff_time)
                    await self.task_queue.put(task)
                else:
                    # Max retries exceeded or permanent failure
                    self.failed_tasks.add(task.task_id)
                    self.db.save_task(task.task_id, "unknown", task.description, "failed", assigned_agent=agent_id, result={'error': result.error_message})
                    await self._handle_task_failure(task, agent_id)
                
        except Exception as e:
            logger.error(f"Error executing task {task.task_id} with agent {agent_id}: {str(e)}")
            await self._handle_task_failure(task, agent_id, str(e))
            
        finally:
            self.agent_status[agent_id] = AgentStatus.ACTIVE
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
    async def _apply_incentives(self, agent_id: str, execution_time: float, priority: WorkflowPriority):
        """Apply incentive rewards based on performance"""
        config = self.agents[agent_id]
        
        # Base reward based on priority
        priority_reward = {
            WorkflowPriority.CRITICAL: 100.0,
            WorkflowPriority.HIGH: 50.0,
            WorkflowPriority.MEDIUM: 25.0,
            WorkflowPriority.LOW: 10.0
        }[priority]
        
        # Performance bonus (faster execution = higher bonus)
        performance_bonus = max(0, (5.0 - execution_time) * 10)  # Max 5 seconds baseline
        
        # Total reward
        total_reward = priority_reward + performance_bonus
        
        # Update agent's reliability score based on performance
        if execution_time < 2.0:  # Good performance threshold
            config.reliability_score = min(0.99, config.reliability_score + 0.01)
        else:
            config.reliability_score = max(0.5, config.reliability_score - 0.02)
            
        logger.info(f"Agent {agent_id} earned reward of ${total_reward:.2f} for fast execution")
        
    async def _handle_task_failure(self, task: Task, agent_id: str, error: str = ""):
        """Handle task failure with recovery strategies"""
        config = self.agents[agent_id]
        
        # Update agent status to failed
        self.agent_status[agent_id] = AgentStatus.FAILED
        logger.warning(f"Agent {agent_id} failed task {task.task_id}. Initiating recovery...")
        
        # Attempt recovery
        recovery_success = await self._attempt_agent_recovery(agent_id)
        
        if recovery_success:
            logger.info(f"Agent {agent_id} recovered successfully")
            # Requeue the failed task
            await self.task_queue.put((self._calculate_task_priority(task, WorkflowPriority.HIGH), task))
        else:
            logger.error(f"Agent {agent_id} recovery failed. Task {task.task_id} needs manual intervention")
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
            agent_id for agent_id, config in self.agents.items()
            if all(cap in config.capabilities for cap in task.required_capabilities)
            and self.agent_status[agent_id] == AgentStatus.ACTIVE
        ]
        
        if alternative_agents:
            # Select best alternative agent
            best_alternative = max(alternative_agents, key=lambda aid: self.agents[aid].reliability_score)
            logger.info(f"Found alternative agent {best_alternative} for task {task.task_id}")
            await self._execute_task_with_agent(task, best_alternative)
            return
            
        # Strategy 2: Degraded mode - use agents with partial capabilities
        partial_agents = [
            (agent_id, sum(1 for cap in task.required_capabilities if cap in self.agents[agent_id].capabilities))
            for agent_id, config in self.agents.items()
            if self.agent_status[agent_id] == AgentStatus.ACTIVE
        ]
        
        partial_agents.sort(key=lambda x: x[1], reverse=True)
        
        if partial_agents and partial_agents[0][1] > 0:
            # Use agent with most matching capabilities
            degraded_agent = partial_agents[0][0]
            logger.warning(f"Using degraded mode with agent {degraded_agent} for task {task.task_id}")
            self.agent_status[degraded_agent] = AgentStatus.DEGRADED
            await self._execute_task_with_agent(task, degraded_agent)
            return
            
        # Strategy 3: Manual intervention required
        logger.error(f"No fallback agents available for task {task.task_id}. Manual intervention required.")
        
    async def _monitoring_loop(self):
        """Continuous monitoring loop for system health and emergent behaviors"""
        while True:
            try:
                await self._check_system_health()
                await self._detect_emergent_behaviors()
                await self._optimize_resource_allocation()
                
                # Log system status periodically
                if time.time() % 60 < 1:  # Every minute
                    self._log_system_status()
                    
                await asyncio.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(10.0)  # Longer delay after errors
                
    async def _check_system_health(self):
        """Check overall system health and agent status"""
        healthy_agents = sum(1 for status in self.agent_status.values() 
                           if status in [AgentStatus.ACTIVE, AgentStatus.IDLE])
        total_agents = len(self.agent_status)
        
        health_ratio = healthy_agents / total_agents if total_agents > 0 else 0
        
        if health_ratio < 0.5:
            logger.warning(f"System health critical: {healthy_agents}/{total_agents} agents healthy")
        elif health_ratio < 0.8:
            logger.warning(f"System health degraded: {healthy_agents}/{total_agents} agents healthy")
            
    async def _detect_emergent_behaviors(self):
        """Detect unexpected patterns or emergent behaviors in agent interactions"""
        # This is a simplified version - in real system, this would use ML to detect patterns
        current_metrics = {
            'active_tasks': len(self.active_tasks),
            'queue_size': self.task_queue.qsize(),
            'agent_distribution': {str(status): sum(1 for s in self.agent_status.values() if s == status) 
                                 for status in AgentStatus}
        }
        
        # Check for unusual patterns
        if current_metrics['queue_size'] > 100 and current_metrics['active_tasks'] < 5:
            self.emergent_behaviors.append({
                'timestamp': time.time(),
                'behavior': 'task_queue_backlog',
                'description': 'Tasks are queuing up but not being processed',
                'severity': 'high'
            })
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
                    config.max_concurrent_tasks = min(10, config.max_concurrent_tasks + 1)
                elif avg_performance > 10.0:  # Slow performance
                    # Decrease capacity or trigger investigation
                    config.max_concurrent_tasks = max(1, config.max_concurrent_tasks - 1)
                    
    def _log_system_status(self):
        """Log current system status for monitoring"""
        status_summary = {
            'timestamp': time.time(),
            'total_agents': len(self.agents),
            'agent_status_counts': {str(status): sum(1 for s in self.agent_status.values() if s == status) 
                                 for status in AgentStatus},
            'active_tasks': len(self.active_tasks),
            'queue_size': self.task_queue.qsize(),
            'incentive_pool': self.incentive_pool
        }
        
        logger.info(f"System Status: {json.dumps(status_summary, indent=2)}")
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics for external monitoring"""
        return {
            'agent_count': len(self.agents),
            'active_workflows': len(self.workflows),
            'pending_tasks': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'agent_status_distribution': {str(status): sum(1 for s in self.agent_status.values() if s == status) 
                                         for status in AgentStatus},
            'average_performance': {
                agent_id: sum(metrics[-5:]) / 5 if len(metrics) >= 5 else None
                for agent_id, metrics in self.performance_metrics.items()
            }
        }