import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.engine import AgentEngine
from main import AutonomousAgentEcosystem


@pytest.mark.asyncio
async def test_engine_drains_inflight_tasks_on_shutdown():
    engine = AgentEngine(enable_advanced_systems=False)
    engine._shutdown_event = asyncio.Event()

    mock_db = MagicMock()
    mock_db.async_init = AsyncMock()
    mock_db.save_agent_async = AsyncMock()
    mock_db.save_workflow_async = AsyncMock()
    mock_db.save_task_async = AsyncMock()
    mock_db.log_metric_async = AsyncMock()
    engine.db = mock_db

    task_id = "task-1"
    await engine._task_queue.mark_active(task_id, {"status": "running"})

    async def complete_task() -> None:
        await asyncio.sleep(0.1)
        await engine._task_queue.mark_completed(task_id)

    asyncio.create_task(complete_task())

    await engine.shutdown()

    assert engine._shutdown_requested is True
    assert engine._shutdown_event.is_set()
    assert engine._task_queue.active_count == 0


@pytest.mark.asyncio
async def test_start_system_respects_shutdown_event():
    shutdown_event = asyncio.Event()
    ecosystem = AutonomousAgentEcosystem(config={})

    ecosystem.initialize_agents = AsyncMock()
    ecosystem.create_sample_workflows = AsyncMock()

    async def fake_engine_start(event: asyncio.Event):
        await event.wait()

    ecosystem.engine.start_engine = AsyncMock(side_effect=fake_engine_start)

    async def fake_monitoring() -> None:
        await shutdown_event.wait()

    ecosystem.dashboard.start_monitoring = AsyncMock(side_effect=fake_monitoring)
    ecosystem.dashboard.request_shutdown = MagicMock()

    start_task = asyncio.create_task(ecosystem.start_system(shutdown_event))
    await asyncio.sleep(0.05)
    shutdown_event.set()
    await start_task

    ecosystem.dashboard.request_shutdown.assert_called_once()
    ecosystem.engine.start_engine.assert_awaited()
    ecosystem.dashboard.start_monitoring.assert_awaited()
