"""
Agents Router - API endpoints for agent management
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/agents", tags=["agents"])

# Module-level state (set by init_router)
_app_state = None


class AgentResponse(BaseModel):
    id: str
    name: str
    icon: str
    status: str
    description: str
    capabilities: List[str]


def init_router(app_state):
    """Initialize router with app state."""
    global _app_state
    _app_state = app_state


@router.get("", response_model=List[AgentResponse])
async def get_agents():
    """Get all registered agents."""
    if not _app_state:
        return []

    agents = []
    agent_meta = {
        "research_agent_001": (
            "🔬",
            "Research · 001",
            "Deepcrawl / Contextual synthesis",
        ),
        "code_agent_001": ("💻", "Code · 001", "Python · QA harness"),
        "filesystem_agent_001": ("📁", "File · 001", "Structured storage / Render"),
    }

    for agent_id, agent in _app_state.agents.items():
        icon, name, desc = agent_meta.get(agent_id, ("🤖", agent_id, "Agent"))
        status = (
            _app_state.engine.agent_status.get(agent_id) if _app_state.engine else None
        )

        agents.append(
            AgentResponse(
                id=agent_id,
                name=name,
                icon=icon,
                status=status.value if status else "idle",
                description=desc,
                capabilities=(
                    [c.value for c in agent.capabilities]
                    if hasattr(agent, "capabilities")
                    else []
                ),
            )
        )

    return agents


@router.get("/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get status of a specific agent."""
    if not _app_state or agent_id not in _app_state.agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = _app_state.agents[agent_id]
    status = _app_state.engine.agent_status.get(agent_id) if _app_state.engine else None

    return {
        "id": agent_id,
        "status": status.value if status else "idle",
        "health": await agent.health_check() if hasattr(agent, "health_check") else {},
    }
