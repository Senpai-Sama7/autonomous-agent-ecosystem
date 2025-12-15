import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.base_agent import AgentContext
import agents.code_agent as code_agent_module


def make_context() -> AgentContext:
    return AgentContext(
        agent_id="code_agent_test",
        current_time=0.0,
        system_load=0.0,
        available_resources={},
        other_agents=[],
        active_workflows=[],
        incentive_pool=0.0,
    )


@pytest.mark.asyncio
async def test_code_agent_initializes_without_docker(monkeypatch):
    monkeypatch.setattr(code_agent_module, "HAS_DOCKER", False, raising=False)
    agent = code_agent_module.CodeAgent("code_agent_001", {})
    context = make_context()
    task = {"code_task_type": "execute_code", "code": "print('hello')"}
    result = await agent.execute_task(task, context)

    assert not result.success
    assert "Docker" in (result.error_message or "")
