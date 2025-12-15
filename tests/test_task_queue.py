import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.task_queue import TaskQueue, WorkflowPriority


@pytest.mark.asyncio
async def test_dependencies_block_until_completed():
    queue = TaskQueue()
    blocker = {"task_id": "blocker"}
    dependent = {"task_id": "dependent", "dependencies": ["blocker"]}

    await queue.enqueue(blocker, 0.1)
    await queue.enqueue(dependent, 0.2)

    # simulate completing blocker after check
    _, task = await queue.dequeue()
    assert task["task_id"] == "blocker"
    await queue.mark_completed("blocker")

    # dependent should now be allowed
    priority, task = await queue.dequeue()
    assert task["task_id"] == "dependent"
    assert queue.dependencies_met(task.get("dependencies", []))


def test_calculate_priority_handles_dependencies_and_deadlines():
    score_with_dep = TaskQueue.calculate_priority(
        WorkflowPriority.MEDIUM, deadline=None, has_dependencies=True
    )
    score_without_dep = TaskQueue.calculate_priority(
        WorkflowPriority.MEDIUM, deadline=None, has_dependencies=False
    )

    assert score_with_dep < score_without_dep
