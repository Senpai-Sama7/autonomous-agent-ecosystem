"""
TaskQueue - Thread-safe priority task queue with dependency tracking.
Extracted from engine.py for better separation of concerns.
"""

import asyncio
import time
from typing import Dict, Set, Optional, Any
from enum import Enum


class WorkflowPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskQueue:
    """Thread-safe async priority queue with task lifecycle management."""

    def __init__(self):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._completed: Set[str] = set()
        self._failed: Set[str] = set()
        self._active: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def enqueue(self, task: Any, priority_score: float):
        """Add task to queue with priority (lower = higher priority)."""
        await self._queue.put((priority_score, task))

    async def dequeue(self):
        """Get next task from queue. Returns (priority_score, task)."""
        return await self._queue.get()

    async def requeue(self, task: Any, priority_score: float):
        """Re-add task with adjusted priority."""
        await self._queue.put((priority_score, task))

    async def mark_active(self, task_id: str, task: Any):
        """Mark task as actively being processed."""
        async with self._lock:
            self._active[task_id] = task

    async def mark_completed(self, task_id: str):
        """Mark task as successfully completed."""
        async with self._lock:
            self._completed.add(task_id)
            self._active.pop(task_id, None)

    async def mark_failed(self, task_id: str):
        """Mark task as failed."""
        async with self._lock:
            self._failed.add(task_id)
            self._active.pop(task_id, None)

    async def remove_active(self, task_id: str):
        """Remove task from active set without marking complete/failed."""
        async with self._lock:
            self._active.pop(task_id, None)

    def is_completed(self, task_id: str) -> bool:
        return task_id in self._completed

    def is_failed(self, task_id: str) -> bool:
        return task_id in self._failed

    def dependencies_met(self, dependencies: list) -> bool:
        """Check if all dependencies are completed."""
        if not dependencies:
            return True
        for dep_id in dependencies:
            if dep_id not in self._completed:
                return dep_id in self._failed  # Failed dep = unmet
        return True

    def has_failed_dependency(self, dependencies: list) -> bool:
        """Check if any dependency has failed."""
        return any(dep_id in self._failed for dep_id in (dependencies or []))

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def completed_tasks(self) -> Set[str]:
        return self._completed.copy()

    @property
    def failed_tasks(self) -> Set[str]:
        return self._failed.copy()

    @property
    def active_tasks(self) -> Dict[str, Any]:
        return self._active.copy()

    @staticmethod
    def calculate_priority(
        base_priority: WorkflowPriority,
        deadline: Optional[float] = None,
        has_dependencies: bool = False,
    ) -> float:
        """Calculate dynamic priority score (lower = higher priority)."""
        base = {
            WorkflowPriority.CRITICAL: 0.1,
            WorkflowPriority.HIGH: 0.3,
            WorkflowPriority.MEDIUM: 0.5,
            WorkflowPriority.LOW: 0.8,
        }.get(base_priority, 0.5)

        # Deadline factor
        if deadline:
            remaining = deadline - time.time()
            if remaining < 0:
                deadline_factor = 0.1
            elif remaining < 60:
                deadline_factor = 0.3
            elif remaining < 300:
                deadline_factor = 0.5
            else:
                deadline_factor = 0.8
        else:
            deadline_factor = 0.7

        dependency_factor = 0.5 if has_dependencies else 0.8
        return base * deadline_factor * dependency_factor
