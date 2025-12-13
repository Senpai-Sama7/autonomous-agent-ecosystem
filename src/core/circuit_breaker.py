"""
Circuit Breaker Pattern Implementation

Prevents cascade failures when external services (LLM, web search) are unavailable.
Implements the standard circuit breaker states: CLOSED, OPEN, HALF_OPEN.
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any, Optional, TypeVar, Generic
from functools import wraps

from src.utils.structured_logger import get_logger
from src.monitoring.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing, requests are rejected immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes in half-open to close
    timeout_seconds: float = 30.0       # Time before half-open attempt
    half_open_max_calls: int = 3        # Max concurrent calls in half-open


class CircuitBreakerError(Exception):
    """Raised when circuit is open and request is rejected."""
    def __init__(self, service_name: str, retry_after: float):
        self.service_name = service_name
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker open for {service_name}. Retry after {retry_after:.1f}s")


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for protecting external service calls.
    
    Usage:
        breaker = CircuitBreaker("llm_api")
        
        async def call_llm():
            async with breaker:
                return await llm_client.complete(...)
    """
    name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    
    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN
    
    async def __aenter__(self):
        """Check if request should be allowed."""
        async with self._lock:
            await self._check_state_transition()
            
            if self._state == CircuitState.OPEN:
                retry_after = self.config.timeout_seconds - (time.time() - self._last_failure_time)
                raise CircuitBreakerError(self.name, max(0, retry_after))
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerError(self.name, 1.0)
                self._half_open_calls += 1
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Record success or failure."""
        async with self._lock:
            if exc_type is None:
                await self._record_success()
            else:
                await self._record_failure()
        return False  # Don't suppress exceptions
    
    async def _check_state_transition(self):
        """Check if state should transition based on timeout."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.config.timeout_seconds:
                logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN (timeout elapsed)")
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._success_count = 0
    
    async def _record_success(self):
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            self._half_open_calls = max(0, self._half_open_calls - 1)
            
            if self._success_count >= self.config.success_threshold:
                logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED (recovered)")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
        
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = max(0, self._failure_count - 1)
    
    async def _record_failure(self):
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        metrics.record_error(f"circuit_breaker_{self.name}")
        
        if self._state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN (failure during recovery)")
            self._state = CircuitState.OPEN
            self._half_open_calls = 0
        
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                logger.warning(f"Circuit {self.name}: CLOSED -> OPEN (threshold reached)")
                self._state = CircuitState.OPEN
    
    def get_stats(self) -> dict:
        """Get current circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": self._last_failure_time,
        }


# Global circuit breakers for common services
_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker for a service."""
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name, config or CircuitBreakerConfig())
    return _breakers[name]


def circuit_protected(service_name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator to protect async functions with circuit breaker.
    
    Usage:
        @circuit_protected("llm_api")
        async def call_llm(prompt: str) -> str:
            return await client.complete(prompt)
    """
    def decorator(func: Callable) -> Callable:
        breaker = get_circuit_breaker(service_name, config)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with breaker:
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator
