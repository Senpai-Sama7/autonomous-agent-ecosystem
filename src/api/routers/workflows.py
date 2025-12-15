"""
Workflows Router - API endpoints for workflow management
"""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/workflows", tags=["workflows"])

# Module-level state
_app_state = None
_manager = None
_workflow_priority = None
_task_class = None
_workflow_class = None


class WorkflowCreateRequest(BaseModel):
    name: str
    description: str = ""
    priority: str = "medium"
    tasks: List[dict] = []


def init_router(app_state, manager, WorkflowPriority, Task, Workflow):
    """Initialize router with dependencies."""
    global _app_state, _manager, _workflow_priority, _task_class, _workflow_class
    _app_state = app_state
    _manager = manager
    _workflow_priority = WorkflowPriority
    _task_class = Task
    _workflow_class = Workflow


@router.get("")
async def get_workflows():
    """Get all workflows."""
    if not _app_state or not _app_state.engine:
        return []

    workflows = []
    for wf_id, workflow in _app_state.engine.workflows.items():
        completed = sum(
            1 for t in workflow.tasks if t.task_id in _app_state.engine.completed_tasks
        )
        total = len(workflow.tasks)

        workflows.append(
            {
                "id": wf_id,
                "name": workflow.name,
                "description": "",
                "status": (
                    "completed"
                    if completed == total
                    else "running" if completed > 0 else "pending"
                ),
                "priority": workflow.priority.value,
                "progress": int((completed / total) * 100) if total > 0 else 0,
                "tasks": [
                    {"id": t.task_id, "description": t.description}
                    for t in workflow.tasks
                ],
            }
        )

    return workflows


@router.post("")
async def create_workflow(request: WorkflowCreateRequest):
    """Create a new workflow."""
    if not _app_state or not _app_state.engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"

    priority_map = {
        "low": _workflow_priority.LOW,
        "medium": _workflow_priority.MEDIUM,
        "high": _workflow_priority.HIGH,
        "critical": _workflow_priority.CRITICAL,
    }

    tasks = []
    for i, task_data in enumerate(request.tasks):
        tasks.append(
            _task_class(
                task_id=f"{workflow_id}_task_{i}",
                description=task_data.get("description", f"Task {i+1}"),
                required_capabilities=task_data.get("capabilities", []),
                priority=priority_map.get(request.priority, _workflow_priority.MEDIUM),
            )
        )

    workflow = _workflow_class(
        workflow_id=workflow_id,
        name=request.name,
        tasks=tasks,
        priority=priority_map.get(request.priority, _workflow_priority.MEDIUM),
    )

    await _app_state.engine.submit_workflow(workflow)

    if _manager:
        await _manager.broadcast(
            "log",
            {
                "timestamp": datetime.now().strftime("%H:%M"),
                "type": "workflow",
                "title": "Workflow Created",
                "message": f"Created workflow: {request.name}",
            },
        )

    return {"id": workflow_id, "name": request.name, "status": "created"}


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get a specific workflow."""
    if not _app_state or workflow_id not in _app_state.engine.workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow = _app_state.engine.workflows[workflow_id]
    completed = sum(
        1 for t in workflow.tasks if t.task_id in _app_state.engine.completed_tasks
    )

    return {
        "id": workflow_id,
        "name": workflow.name,
        "status": "completed" if completed == len(workflow.tasks) else "running",
        "progress": (
            int((completed / len(workflow.tasks)) * 100) if workflow.tasks else 0
        ),
        "tasks": [
            {
                "id": t.task_id,
                "description": t.description,
                "status": (
                    "completed"
                    if t.task_id in _app_state.engine.completed_tasks
                    else "pending"
                ),
            }
            for t in workflow.tasks
        ],
    }


@router.post("/{workflow_id}/run")
async def run_workflow(workflow_id: str):
    """Run/resume a workflow."""
    if not _app_state or workflow_id not in _app_state.engine.workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow = _app_state.engine.workflows[workflow_id]

    for task in workflow.tasks:
        if task.task_id not in _app_state.engine.completed_tasks:
            await _app_state.engine._queue_task(task, workflow.priority)

    return {"status": "running", "message": f"Workflow {workflow_id} is now running"}
