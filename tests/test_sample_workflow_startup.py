import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import AutonomousAgentEcosystem


def _build_sample_workflow_file(directory: Path) -> Path:
    workflow_content = {
        "test_workflow": {
            "description": "Sample workflow for testing",
            "priority": "low",
            "budget": 5.0,
            "tasks": [
                {
                    "id": "read",
                    "capability": "file_operations",
                    "instruction": "Read a file",
                    "payload": {"operation": "read_file", "path": "/tmp/example.txt"},
                }
            ],
        }
    }

    file_path = directory / "sample.yaml"
    file_path.write_text(json.dumps(workflow_content))
    return file_path


def _prepare_ecosystem(config: Dict[str, Any]) -> AutonomousAgentEcosystem:
    ecosystem = AutonomousAgentEcosystem(config=config)
    ecosystem.initialize_agents = AsyncMock()
    ecosystem.engine.start_engine = AsyncMock()
    ecosystem.dashboard.start_monitoring = AsyncMock()
    ecosystem.engine.submit_workflow = AsyncMock()
    ecosystem.generate_final_report = AsyncMock()
    ecosystem.dashboard.get_system_health = lambda: {
        "health_score": 1.0,
        "status": "ok",
        "active_agents_count": 0,
    }
    return ecosystem


@pytest.mark.asyncio
async def test_startup_with_sample_workflows(tmp_path: Path):
    workflow_file = _build_sample_workflow_file(tmp_path)

    ecosystem = _prepare_ecosystem({"duration": 0, "sample_workflows": str(workflow_file)})
    await ecosystem.start_system()
    ecosystem.engine.submit_workflow.assert_awaited()


@pytest.mark.asyncio
async def test_startup_without_sample_workflows(tmp_path: Path):
    missing_file = tmp_path / "missing.yaml"

    ecosystem = _prepare_ecosystem({"duration": 0, "sample_workflows": str(missing_file)})
    await ecosystem.start_system()
    ecosystem.engine.submit_workflow.assert_not_awaited()
