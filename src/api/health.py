"""
Health Check System - Kubernetes-compatible probes
"""

import asyncio
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Response, status

router = APIRouter(prefix="/health", tags=["health"])


@dataclass
class HealthStatus:
    status: str  # healthy, unhealthy, degraded
    message: str = ""
    response_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """Centralized health checking."""

    def __init__(self):
        self.agent_engine = None
        self.db = None
        self.start_time = time.time()
        self._initialized = False

    def set_components(self, agent_engine=None, db_manager=None):
        self.agent_engine = agent_engine
        self.db = db_manager
        self._initialized = True

    async def check_database(self) -> HealthStatus:
        start = time.time()
        try:
            if not self.db:
                return HealthStatus("unhealthy", "Database not initialized")

            # Quick connectivity check
            if hasattr(self.db, "get_database_stats"):
                await asyncio.wait_for(
                    asyncio.to_thread(self.db.get_database_stats), timeout=5
                )

            ms = (time.time() - start) * 1000
            return HealthStatus(
                "degraded" if ms > 100 else "healthy",
                f"DB {'slow' if ms > 100 else 'ok'} ({ms:.0f}ms)",
                ms,
            )
        except Exception as e:
            return HealthStatus(
                "unhealthy", f"DB error: {e}", (time.time() - start) * 1000
            )

    async def check_agents(self) -> HealthStatus:
        start = time.time()
        try:
            if not self.agent_engine:
                return HealthStatus("unhealthy", "Engine not initialized")

            statuses = getattr(self.agent_engine, "agent_status", {})
            total = len(statuses)
            healthy = sum(
                1
                for s in statuses.values()
                if getattr(s, "value", s) in ["idle", "active"]
            )

            ms = (time.time() - start) * 1000
            if healthy == 0:
                return HealthStatus(
                    "unhealthy", "No healthy agents", ms, {"total": total}
                )
            elif healthy < total * 0.5:
                return HealthStatus("degraded", f"{healthy}/{total} agents healthy", ms)
            return HealthStatus("healthy", f"{healthy}/{total} agents healthy", ms)
        except Exception as e:
            return HealthStatus("unhealthy", f"Agent check error: {e}")

    async def check_docker(self) -> HealthStatus:
        start = time.time()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run, ["docker", "info"], capture_output=True, timeout=5
                ),
                timeout=6,
            )
            ms = (time.time() - start) * 1000
            return HealthStatus(
                "healthy" if result.returncode == 0 else "unhealthy",
                (
                    "Docker available"
                    if result.returncode == 0
                    else "Docker not responding"
                ),
                ms,
            )
        except FileNotFoundError:
            return HealthStatus("unhealthy", "Docker not installed")
        except Exception as e:
            return HealthStatus("unhealthy", f"Docker error: {e}")

    @property
    def uptime(self) -> float:
        return time.time() - self.start_time


_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    global _checker
    if _checker is None:
        _checker = HealthChecker()
    return _checker


def init_health_checker(agent_engine=None, db_manager=None) -> HealthChecker:
    checker = get_health_checker()
    checker.set_components(agent_engine, db_manager)
    return checker


@router.get("/live")
async def liveness():
    """Kubernetes liveness probe - is process alive?"""
    return {
        "status": "alive",
        "uptime_seconds": get_health_checker().uptime,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/ready")
async def readiness(response: Response):
    """Kubernetes readiness probe - ready to serve traffic?"""
    checker = get_health_checker()

    results = await asyncio.gather(
        checker.check_database(),
        checker.check_agents(),
        checker.check_docker(),
        return_exceptions=True,
    )

    checks = {}
    for name, r in zip(["database", "agents", "docker"], results):
        checks[name] = (
            asdict(r)
            if isinstance(r, HealthStatus)
            else {"status": "unhealthy", "message": str(r)}
        )

    all_healthy = all(c.get("status") == "healthy" for c in checks.values())

    if not all_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks,
        "uptime_seconds": checker.uptime,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/startup")
async def startup(response: Response):
    """Kubernetes startup probe - initial startup complete?"""
    checker = get_health_checker()

    if not checker._initialized:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "starting", "message": "Not yet initialized"}

    return {
        "status": "started",
        "uptime_seconds": checker.uptime,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("")
@router.get("/")
async def health_summary():
    """Overall health summary."""
    checker = get_health_checker()

    results = await asyncio.gather(
        checker.check_database(),
        checker.check_agents(),
        checker.check_docker(),
        return_exceptions=True,
    )

    checks = {}
    for name, r in zip(["database", "agents", "docker"], results):
        checks[name] = (
            asdict(r)
            if isinstance(r, HealthStatus)
            else {"status": "unhealthy", "message": str(r)}
        )

    overall = (
        "healthy"
        if all(c.get("status") == "healthy" for c in checks.values())
        else "degraded"
    )

    return {
        "overall_status": overall,
        "checks": checks,
        "uptime_seconds": checker.uptime,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
