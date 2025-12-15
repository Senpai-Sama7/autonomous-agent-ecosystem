import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.engine import AgentEngine, AgentConfig, AgentStatus, WorkflowPriority, Task


@pytest.mark.asyncio
async def test_agent_concurrency_limit_blocks_additional_tasks():
    engine = AgentEngine()

    async def noop(*args, **kwargs):
        return None

    engine.db.save_agent_async = noop  # skip DB writes

    await engine.register_agent(
        AgentConfig(
            agent_id="agent_1",
            capabilities=["work"],
            max_concurrent_tasks=1,
        )
    )

    engine.agent_status["agent_1"] = AgentStatus.ACTIVE
    engine._agent_active_counts["agent_1"] = 1  # simulate in-flight task

    task = Task(
        task_id="task_1",
        description="test",
        required_capabilities=["work"],
        priority=WorkflowPriority.MEDIUM,
    )

    assert await engine._find_suitable_agent(task) is None


@pytest.mark.asyncio
async def test_agent_concurrency_allows_when_below_limit():
    engine = AgentEngine()

    async def noop(*args, **kwargs):
        return None

    engine.db.save_agent_async = noop

    await engine.register_agent(
        AgentConfig(
            agent_id="agent_1",
            capabilities=["work"],
            max_concurrent_tasks=2,
        )
    )

    engine.agent_status["agent_1"] = AgentStatus.ACTIVE
    engine._agent_active_counts["agent_1"] = 1

    task = Task(
        task_id="task_2",
        description="test",
        required_capabilities=["work"],
        priority=WorkflowPriority.MEDIUM,
    )

    assert await engine._find_suitable_agent(task) == "agent_1"
