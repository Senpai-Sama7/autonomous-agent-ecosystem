import asyncio
import os
import sys
from unittest.mock import AsyncMock

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.base_agent import TaskResult
from src.core.engine import AgentConfig, AgentEngine, AgentUnavailableError, Task, Workflow
from src.core.task_queue import WorkflowPriority


class AvailableDummyAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.is_available = True

    async def execute_task(self, task: dict, context: dict | None = None) -> TaskResult:
        await asyncio.sleep(0)
        return TaskResult(success=True, result_data={"task": task.get("task_id"), "agent": self.agent_id})


@pytest.fixture
def engine() -> AgentEngine:
    engine_instance = AgentEngine()
    engine_instance.db.save_workflow_async = AsyncMock()
    engine_instance.db.save_task_async = AsyncMock()
    engine_instance.db.save_agent_async = AsyncMock()
    return engine_instance


@pytest.mark.asyncio
async def test_submit_workflow_missing_agent_fails_fast(engine: AgentEngine) -> None:
    workflow = Workflow(
        workflow_id="wf_missing_agent",
        name="missing agent workflow",
        tasks=[
            Task(
                task_id="task_missing",
                description="requires unavailable capability",
                required_capabilities=["web_search"],
                priority=WorkflowPriority.MEDIUM,
            )
        ],
    )

    with pytest.raises(AgentUnavailableError) as excinfo:
        await engine.submit_workflow(workflow)

    assert "web_search" in str(excinfo.value)
    assert "task_missing" in str(excinfo.value)


@pytest.mark.asyncio
async def test_submit_workflow_succeeds_when_agent_available(engine: AgentEngine) -> None:
    agent = AvailableDummyAgent("agent_web")
    await engine.register_agent(
        AgentConfig(
            agent_id="agent_web",
            capabilities=["web_search"],
            reliability_score=0.9,
        ),
        instance=agent,
    )

    workflow = Workflow(
        workflow_id="wf_available_agent",
        name="available agent workflow",
        tasks=[
            Task(
                task_id="task_web",
                description="requires available capability",
                required_capabilities=["web_search"],
                priority=WorkflowPriority.MEDIUM,
            )
        ],
    )

    await engine.submit_workflow(workflow)

    assert workflow.workflow_id in engine.workflows
    assert engine._task_queue._queue.qsize() == 1
