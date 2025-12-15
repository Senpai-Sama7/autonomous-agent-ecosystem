"""
System routes for ASTRO API.
Handles system status, start/stop, health checks.
"""

import asyncio
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/system", tags=["system"])

# These will be set by server.py during app initialization
_app_state = None
_manager = None


def init_router(app_state: Any, manager: Any):
    """Initialize router with app state and connection manager."""
    global _app_state, _manager
    _app_state = app_state
    _manager = manager


class SystemStatusResponse(BaseModel):
    status: str
    agents_count: int
    active_workflows: int
    uptime: float
    version: str = "1.0.0"


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get current system status."""
    return SystemStatusResponse(
        status="online" if _app_state.running else "ready",
        agents_count=len(_app_state.agents),
        active_workflows=len(_app_state.engine.workflows) if _app_state.engine else 0,
        uptime=time.time() - _app_state.start_time,
    )


@router.post("/start")
async def start_system():
    """Start the agent system."""
    if _app_state.running:
        return {"status": "already_running", "message": "System is already running"}

    _app_state.running = True

    # Initialize NL interface if needed
    if _app_state.llm_client and not _app_state.nl_interface:
        from src.core.nl_interface import NaturalLanguageInterface

        model = getattr(_app_state, "llm_model", "llama3.2")
        _app_state.nl_interface = NaturalLanguageInterface(
            engine=_app_state.engine, llm_client=_app_state.llm_client, model_name=model
        )

    # Start engine in background
    asyncio.create_task(_app_state.engine.start_engine())

    await _manager.broadcast("system_status", {"status": "online"})
    await _manager.broadcast(
        "log",
        {
            "timestamp": datetime.now().strftime("%H:%M"),
            "type": "system",
            "title": "System Started",
            "message": "ASTRO agent ecosystem is now online.",
        },
    )

    return {"status": "started", "message": "System started successfully"}


@router.post("/stop")
async def stop_system():
    """Stop the agent system."""
    if not _app_state.running:
        return {"status": "not_running", "message": "System is not running"}

    _app_state.running = False
    await _app_state.engine.shutdown()
    await _manager.broadcast("system_status", {"status": "offline"})

    return {"status": "stopped", "message": "System stopped successfully"}
