import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.base_agent import AgentContext
from agents.git_agent import GitAgent
from agents.test_agent import TestAgent
from agents.analysis_agent import AnalysisAgent


def make_context(agent_id: str) -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        current_time=0.0,
        system_load=0.0,
        available_resources={},
        other_agents=[],
        active_workflows=[],
        incentive_pool=0.0,
    )


@pytest.mark.asyncio
async def test_git_agent_runs_status_quickly():
    agent = GitAgent("git_agent_test", {})
    task = {"operation": "status", "arguments": ["--short"]}
    result = await agent.execute_task(task, make_context("git_agent_test"))
    assert result.success
    assert "output" in (result.result_data or {})


@pytest.mark.asyncio
async def test_test_agent_executes_python_command():
    agent = TestAgent("test_agent_test", {"command_timeout": 10})
    task = {"command": [sys.executable, "-c", "print('ok')"]}
    result = await agent.execute_task(task, make_context("test_agent_test"))
    assert result.success
    assert "ok" in (result.result_data or {}).get("output", "")


@pytest.mark.asyncio
async def test_analysis_agent_supports_arguments():
    agent = AnalysisAgent("analysis_agent_test", {"command_timeout": 10})
    task = {
        "tool": sys.executable,
        "path": "-c",
        "arguments": ["print('analysis')"],
    }
    result = await agent.execute_task(task, make_context("analysis_agent_test"))
    assert result.success
    assert "analysis" in (result.result_data or {}).get("output", "")
