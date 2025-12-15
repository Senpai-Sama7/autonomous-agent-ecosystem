"""
Self-Healing System
Enterprise-grade fault detection, automatic recovery, and resilience patterns.
"""

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Generic
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import statistics
import json
import hashlib

logger = logging.getLogger(__name__)

T = TypeVar("T")


class HealthStatus(Enum):
    """Component health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    RECOVERING = "recovering"


class FailureType(Enum):
    """Types of failures"""

    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    DATA_CORRUPTION = "data_corruption"
    NETWORK_ERROR = "network_error"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies"""

    RETRY = "retry"
    RESTART = "restart"
    FAILOVER = "failover"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADE = "graceful_degrade"
    ESCALATE = "escalate"
    IGNORE = "ignore"


@dataclass
class HealthCheck:
    """Health check result"""

    component_id: str
    status: HealthStatus
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Failure:
    """Failure record"""

    failure_id: str
    component_id: str
    failure_type: FailureType
    timestamp: str
    error_message: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recovered: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_time_ms: Optional[float] = None


@dataclass
class CircuitBreakerState:
    """Circuit breaker state"""

    failures: int = 0
    successes: int = 0
    last_failure_time: Optional[float] = None
    state: str = "closed"  # closed, open, half-open
    next_retry_time: Optional[float] = None


class CircuitBreaker:
    """Circuit breaker pattern implementation"""

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout_seconds: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls
        self._state = CircuitBreakerState()
        self._half_open_calls = 0

    @property
    def is_open(self) -> bool:
        if self._state.state == "open":
            if time.time() >= (self._state.next_retry_time or 0):
                self._state.state = "half-open"
                self._half_open_calls = 0
                return False
            return True
        return False

    @property
    def is_closed(self) -> bool:
        return self._state.state == "closed"

    def record_success(self):
        """Record a successful call"""
        if self._state.state == "half-open":
            self._state.successes += 1
            if self._state.successes >= self.success_threshold:
                self._reset()
        else:
            self._state.successes += 1

    def record_failure(self):
        """Record a failed call"""
        self._state.failures += 1
        self._state.last_failure_time = time.time()

        if self._state.state == "half-open":
            self._trip()
        elif self._state.failures >= self.failure_threshold:
            self._trip()

    def _trip(self):
        """Trip the circuit breaker"""
        self._state.state = "open"
        self._state.next_retry_time = time.time() + self.timeout_seconds
        logger.warning(
            f"Circuit breaker tripped. Will retry at {self._state.next_retry_time}"
        )

    def _reset(self):
        """Reset the circuit breaker"""
        self._state = CircuitBreakerState()
        logger.info("Circuit breaker reset to closed state")

    def can_execute(self) -> bool:
        """Check if a call can be executed"""
        if self.is_closed:
            return True
        if self._state.state == "half-open":
            return self._half_open_calls < self.half_open_max_calls
        return not self.is_open


class RetryPolicy:
    """Configurable retry policy with exponential backoff"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        delay = min(self.base_delay * (self.exponential_base**attempt), self.max_delay)
        if self.jitter:
            import random

            delay *= 0.5 + random.random()
        return delay

    async def execute_with_retry(
        self, func: Callable, *args, **kwargs
    ) -> Tuple[bool, Any, List[Exception]]:
        """Execute function with retry logic"""
        exceptions = []

        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return True, result, exceptions
            except Exception as e:
                exceptions.append(e)
                if attempt < self.max_retries:
                    delay = self.get_delay(attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} after {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)

        return False, None, exceptions


class HealthMonitor:
    """Monitors component health and detects anomalies"""

    def __init__(
        self,
        check_interval: float = 10.0,
        history_size: int = 100,
        degraded_threshold: float = 0.7,
        unhealthy_threshold: float = 0.5,
    ):
        self.check_interval = check_interval
        self.history_size = history_size
        self.degraded_threshold = degraded_threshold
        self.unhealthy_threshold = unhealthy_threshold

        self._health_checks: Dict[str, Callable] = {}
        self._health_history: Dict[str, deque] = {}
        self._latency_history: Dict[str, deque] = {}
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

    def register_check(self, component_id: str, check_func: Callable):
        """Register a health check function"""
        self._health_checks[component_id] = check_func
        self._health_history[component_id] = deque(maxlen=self.history_size)
        self._latency_history[component_id] = deque(maxlen=self.history_size)

    async def start(self):
        """Start health monitoring"""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health Monitor started")

    async def stop(self):
        """Stop health monitoring"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health Monitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            for component_id, check_func in self._health_checks.items():
                try:
                    result = await self._run_check(component_id, check_func)
                    self._health_history[component_id].append(result)
                except Exception as e:
                    logger.error(f"Health check failed for {component_id}: {e}")

            await asyncio.sleep(self.check_interval)

    async def _run_check(self, component_id: str, check_func: Callable) -> HealthCheck:
        """Run a single health check"""
        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()

            latency = (time.time() - start_time) * 1000
            self._latency_history[component_id].append(latency)

            status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            return HealthCheck(
                component_id=component_id,
                status=status,
                latency_ms=latency,
                message="Check passed" if result else "Check failed",
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return HealthCheck(
                component_id=component_id,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                message=str(e),
            )

    def get_component_health(self, component_id: str) -> HealthStatus:
        """Get current health status for a component"""
        if component_id not in self._health_history:
            return HealthStatus.UNKNOWN

        history = list(self._health_history[component_id])
        if not history:
            return HealthStatus.UNKNOWN

        # Calculate success rate from recent checks
        recent = history[-10:] if len(history) >= 10 else history
        success_rate = sum(1 for h in recent if h.status == HealthStatus.HEALTHY) / len(
            recent
        )

        if success_rate >= self.degraded_threshold:
            return HealthStatus.HEALTHY
        elif success_rate >= self.unhealthy_threshold:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY

    def get_latency_stats(self, component_id: str) -> Dict[str, float]:
        """Get latency statistics for a component"""
        if component_id not in self._latency_history:
            return {}

        latencies = list(self._latency_history[component_id])
        if not latencies:
            return {}

        return {
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": (
                sorted(latencies)[int(len(latencies) * 0.95)]
                if len(latencies) >= 20
                else max(latencies)
            ),
            "stddev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        }


class RecoveryManager:
    """Manages failure recovery strategies"""

    def __init__(self):
        self._strategies: Dict[FailureType, RecoveryStrategy] = {
            FailureType.TIMEOUT: RecoveryStrategy.RETRY,
            FailureType.EXCEPTION: RecoveryStrategy.RETRY,
            FailureType.RESOURCE_EXHAUSTION: RecoveryStrategy.GRACEFUL_DEGRADE,
            FailureType.DEPENDENCY_FAILURE: RecoveryStrategy.CIRCUIT_BREAK,
            FailureType.NETWORK_ERROR: RecoveryStrategy.RETRY,
            FailureType.RATE_LIMIT: RecoveryStrategy.GRACEFUL_DEGRADE,
            FailureType.DATA_CORRUPTION: RecoveryStrategy.ESCALATE,
            FailureType.UNKNOWN: RecoveryStrategy.RETRY,
        }
        self._recovery_handlers: Dict[RecoveryStrategy, Callable] = {}
        self._failure_log: List[Failure] = []

    def register_handler(self, strategy: RecoveryStrategy, handler: Callable):
        """Register a recovery handler"""
        self._recovery_handlers[strategy] = handler

    def set_strategy(self, failure_type: FailureType, strategy: RecoveryStrategy):
        """Set recovery strategy for failure type"""
        self._strategies[failure_type] = strategy

    async def recover(self, failure: Failure) -> bool:
        """Attempt recovery from failure"""
        strategy = self._strategies.get(failure.failure_type, RecoveryStrategy.RETRY)
        failure.recovery_strategy = strategy

        start_time = time.time()

        handler = self._recovery_handlers.get(strategy)
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    success = await handler(failure)
                else:
                    success = handler(failure)

                failure.recovered = success
                failure.recovery_time_ms = (time.time() - start_time) * 1000

                self._failure_log.append(failure)

                if success:
                    logger.info(
                        f"Recovery successful for {failure.component_id} using {strategy.value}"
                    )
                else:
                    logger.warning(f"Recovery failed for {failure.component_id}")

                return success
            except Exception as e:
                logger.error(f"Recovery handler error: {e}")
                return False

        logger.warning(f"No handler for strategy: {strategy.value}")
        return False

    def get_failure_stats(self) -> Dict[str, Any]:
        """Get failure statistics"""
        if not self._failure_log:
            return {"total": 0}

        by_type = {}
        by_component = {}
        recovered = 0

        for failure in self._failure_log:
            by_type[failure.failure_type.value] = (
                by_type.get(failure.failure_type.value, 0) + 1
            )
            by_component[failure.component_id] = (
                by_component.get(failure.component_id, 0) + 1
            )
            if failure.recovered:
                recovered += 1

        return {
            "total": len(self._failure_log),
            "recovered": recovered,
            "recovery_rate": recovered / len(self._failure_log),
            "by_type": by_type,
            "by_component": by_component,
        }


class SelfHealingSystem:
    """Main self-healing system orchestrator"""

    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.recovery_manager = RecoveryManager()
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._retry_policies: Dict[str, RetryPolicy] = {}
        self._component_callbacks: Dict[str, Callable] = {}
        self._running = False
        self._healing_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the self-healing system"""
        await self.health_monitor.start()
        self._running = True
        self._healing_task = asyncio.create_task(self._healing_loop())
        logger.info("Self-Healing System started")

    async def stop(self):
        """Stop the self-healing system"""
        self._running = False
        if self._healing_task:
            self._healing_task.cancel()
            try:
                await self._healing_task
            except asyncio.CancelledError:
                pass
        await self.health_monitor.stop()
        logger.info("Self-Healing System stopped")

    def register_component(
        self,
        component_id: str,
        health_check: Callable,
        recovery_callback: Optional[Callable] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        """Register a component for self-healing"""
        self.health_monitor.register_check(component_id, health_check)

        if recovery_callback:
            self._component_callbacks[component_id] = recovery_callback

        if circuit_breaker:
            self._circuit_breakers[component_id] = circuit_breaker
        else:
            self._circuit_breakers[component_id] = CircuitBreaker()

        if retry_policy:
            self._retry_policies[component_id] = retry_policy
        else:
            self._retry_policies[component_id] = RetryPolicy()

        logger.info(f"Registered component for self-healing: {component_id}")

    async def _healing_loop(self):
        """Main healing loop"""
        while self._running:
            for component_id in list(self._component_callbacks.keys()):
                status = self.health_monitor.get_component_health(component_id)

                if status in (HealthStatus.UNHEALTHY, HealthStatus.DEGRADED):
                    await self._attempt_healing(component_id, status)

            await asyncio.sleep(5.0)

    async def _attempt_healing(self, component_id: str, status: HealthStatus):
        """Attempt to heal an unhealthy component"""
        cb = self._circuit_breakers.get(component_id)
        if cb and not cb.can_execute():
            logger.debug(f"Circuit breaker open for {component_id}, skipping healing")
            return

        callback = self._component_callbacks.get(component_id)
        if not callback:
            return

        retry_policy = self._retry_policies.get(component_id)
        if retry_policy:
            success, result, exceptions = await retry_policy.execute_with_retry(
                callback
            )

            if success:
                if cb:
                    cb.record_success()
                logger.info(f"Successfully healed component: {component_id}")
            else:
                if cb:
                    cb.record_failure()
                logger.warning(f"Failed to heal component: {component_id}")

    async def execute_with_protection(
        self, component_id: str, func: Callable, *args, **kwargs
    ) -> Tuple[bool, Any]:
        """Execute a function with full self-healing protection"""
        cb = self._circuit_breakers.get(component_id)

        if cb and not cb.can_execute():
            logger.warning(f"Circuit breaker open for {component_id}")
            return False, None

        retry_policy = self._retry_policies.get(component_id, RetryPolicy())
        success, result, exceptions = await retry_policy.execute_with_retry(
            func, *args, **kwargs
        )

        if cb:
            if success:
                cb.record_success()
            else:
                cb.record_failure()

        return success, result

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        component_health = {}
        overall_status = HealthStatus.HEALTHY

        for component_id in self._circuit_breakers.keys():
            status = self.health_monitor.get_component_health(component_id)
            component_health[component_id] = {
                "status": status.value,
                "latency": self.health_monitor.get_latency_stats(component_id),
            }

            if status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif (
                status == HealthStatus.DEGRADED
                and overall_status == HealthStatus.HEALTHY
            ):
                overall_status = HealthStatus.DEGRADED

        return {
            "overall_status": overall_status.value,
            "components": component_health,
            "failure_stats": self.recovery_manager.get_failure_stats(),
            "timestamp": datetime.now().isoformat(),
        }


# Singleton instance
_self_healing: Optional[SelfHealingSystem] = None


def get_self_healing_system() -> SelfHealingSystem:
    """Get or create the self-healing system singleton"""
    global _self_healing
    if _self_healing is None:
        _self_healing = SelfHealingSystem()
    return _self_healing
