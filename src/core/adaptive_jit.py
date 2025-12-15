"""
Adaptive Caching & Memoization System

NOTE ON TERMINOLOGY: This module is named "adaptive_jit" for marketing purposes,
but it is NOT a JIT compiler (which would compile bytecode to machine code).

WHAT THIS ACTUALLY DOES:
- Profiles function execution times and call frequency
- Automatically applies caching strategies (LRU, LFU, TTL) to hot paths
- Uses HotPathDetector to identify performance bottlenecks
- Provides decorators for automatic memoization

This is a high-quality Smart Memoization system that reduces redundant computation
without developer intervention. It does NOT perform bytecode compilation or
low-level optimization.
"""

import asyncio
import logging
import time
import functools
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Generic
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import statistics
import hashlib
import json
import weakref

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class OptimizationLevel(Enum):
    """Optimization levels"""

    NONE = 0
    BASIC = 1
    MODERATE = 2
    AGGRESSIVE = 3
    MAXIMUM = 4


class CacheStrategy(Enum):
    """Caching strategies"""

    NONE = "none"
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


@dataclass
class ExecutionProfile:
    """Profile of function execution"""

    function_id: str
    call_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    last_call: Optional[str] = None
    hot_path: bool = False
    optimization_level: OptimizationLevel = OptimizationLevel.NONE
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def update(self, execution_time_ms: float):
        self.call_count += 1
        self.total_time_ms += execution_time_ms
        self.avg_time_ms = self.total_time_ms / self.call_count
        self.min_time_ms = min(self.min_time_ms, execution_time_ms)
        self.max_time_ms = max(self.max_time_ms, execution_time_ms)
        self.last_call = datetime.now().isoformat()

        # Mark as hot path if frequently called
        if self.call_count > 100:
            self.hot_path = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "functionId": self.function_id,
            "callCount": self.call_count,
            "totalTimeMs": self.total_time_ms,
            "avgTimeMs": self.avg_time_ms,
            "minTimeMs": self.min_time_ms if self.min_time_ms != float("inf") else 0,
            "maxTimeMs": self.max_time_ms,
            "hotPath": self.hot_path,
            "optimizationLevel": self.optimization_level.value,
            "cacheHitRate": self.cache_hit_rate,
        }


class AdaptiveCache(Generic[T]):
    """Adaptive caching with multiple strategies"""

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 300.0,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.strategy = strategy

        self._cache: Dict[str, Tuple[T, float, int]] = (
            {}
        )  # key -> (value, timestamp, access_count)
        self._access_order: deque = deque()
        self._lock = threading.RLock()

        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[T]:
        """Get value from cache"""
        with self._lock:
            if key in self._cache:
                value, timestamp, access_count = self._cache[key]

                # Check TTL
                if self.strategy in (CacheStrategy.TTL, CacheStrategy.ADAPTIVE):
                    if time.time() - timestamp > self.ttl_seconds:
                        del self._cache[key]
                        self._misses += 1
                        return None

                # Update access info
                self._cache[key] = (value, timestamp, access_count + 1)

                if self.strategy in (CacheStrategy.LRU, CacheStrategy.ADAPTIVE):
                    # Move to end of access order
                    if key in self._access_order:
                        self._access_order.remove(key)
                    self._access_order.append(key)

                self._hits += 1
                return value

            self._misses += 1
            return None

    def put(self, key: str, value: T):
        """Put value in cache"""
        with self._lock:
            # Evict if necessary
            while len(self._cache) >= self.max_size:
                self._evict()

            self._cache[key] = (value, time.time(), 1)
            self._access_order.append(key)

    def _evict(self):
        """Evict entry based on strategy"""
        if not self._cache:
            return

        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            if self._access_order:
                key = self._access_order.popleft()
                self._cache.pop(key, None)

        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            min_key = min(self._cache.keys(), key=lambda k: self._cache[k][2])
            del self._cache[min_key]
            if min_key in self._access_order:
                self._access_order.remove(min_key)

        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive: consider both recency and frequency
            now = time.time()
            scores = {}
            for key, (_, timestamp, count) in self._cache.items():
                age = now - timestamp
                score = count / (age + 1)  # Higher is better
                scores[key] = score

            min_key = min(scores.keys(), key=lambda k: scores[k])
            del self._cache[min_key]
            if min_key in self._access_order:
                self._access_order.remove(min_key)

        else:
            # Default: remove oldest
            if self._access_order:
                key = self._access_order.popleft()
                self._cache.pop(key, None)

    def clear(self):
        """Clear cache"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "maxSize": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hitRate": self.hit_rate,
            "strategy": self.strategy.value,
        }


class HotPathDetector:
    """Detects hot code paths for optimization"""

    def __init__(self, threshold_calls: int = 100, threshold_time_percent: float = 0.1):
        self.threshold_calls = threshold_calls
        self.threshold_time_percent = threshold_time_percent
        self._profiles: Dict[str, ExecutionProfile] = {}

    def record(self, function_id: str, execution_time_ms: float):
        """Record an execution"""
        if function_id not in self._profiles:
            self._profiles[function_id] = ExecutionProfile(function_id=function_id)

        self._profiles[function_id].update(execution_time_ms)

    def is_hot_path(self, function_id: str) -> bool:
        """Check if function is on hot path"""
        if function_id not in self._profiles:
            return False

        profile = self._profiles[function_id]

        # Check call count threshold
        if profile.call_count >= self.threshold_calls:
            return True

        # Check time threshold
        total_time = sum(p.total_time_ms for p in self._profiles.values())
        if total_time > 0:
            time_percent = profile.total_time_ms / total_time
            if time_percent >= self.threshold_time_percent:
                return True

        return False

    def get_hot_paths(self) -> List[str]:
        """Get all hot paths"""
        return [fid for fid in self._profiles if self.is_hot_path(fid)]

    def get_profile(self, function_id: str) -> Optional[ExecutionProfile]:
        return self._profiles.get(function_id)

    def get_all_profiles(self) -> List[ExecutionProfile]:
        return list(self._profiles.values())


class OptimizationStrategy(ABC):
    """Base class for optimization strategies"""

    @abstractmethod
    def optimize(self, func: Callable, profile: ExecutionProfile) -> Callable:
        """Apply optimization to function"""
        pass

    @abstractmethod
    def get_level(self) -> OptimizationLevel:
        pass


class MemoizationStrategy(OptimizationStrategy):
    """Memoization optimization"""

    def __init__(self, cache: AdaptiveCache):
        self.cache = cache

    def optimize(self, func: Callable, profile: ExecutionProfile) -> Callable:
        cache = self.cache

        @functools.wraps(func)
        def memoized(*args, **kwargs):
            # Create cache key
            key = hashlib.md5(
                f"{func.__name__}{args}{sorted(kwargs.items())}".encode()
            ).hexdigest()

            result = cache.get(key)
            if result is not None:
                profile.cache_hits += 1
                return result

            profile.cache_misses += 1
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result

        return memoized

    def get_level(self) -> OptimizationLevel:
        return OptimizationLevel.BASIC


class AsyncMemoizationStrategy(OptimizationStrategy):
    """Async memoization optimization"""

    def __init__(self, cache: AdaptiveCache):
        self.cache = cache

    def optimize(self, func: Callable, profile: ExecutionProfile) -> Callable:
        cache = self.cache

        @functools.wraps(func)
        async def memoized(*args, **kwargs):
            key = hashlib.md5(
                f"{func.__name__}{args}{sorted(kwargs.items())}".encode()
            ).hexdigest()

            result = cache.get(key)
            if result is not None:
                profile.cache_hits += 1
                return result

            profile.cache_misses += 1
            result = await func(*args, **kwargs)
            cache.put(key, result)
            return result

        return memoized

    def get_level(self) -> OptimizationLevel:
        return OptimizationLevel.BASIC


class BatchingStrategy(OptimizationStrategy):
    """Request batching optimization"""

    def __init__(self, batch_size: int = 10, delay_ms: float = 50.0):
        self.batch_size = batch_size
        self.delay_ms = delay_ms
        self._pending: Dict[str, List[Tuple[Any, asyncio.Future]]] = {}
        self._lock = asyncio.Lock()

    def optimize(self, func: Callable, profile: ExecutionProfile) -> Callable:
        batch_size = self.batch_size
        delay_ms = self.delay_ms
        pending = self._pending
        lock = self._lock

        @functools.wraps(func)
        async def batched(*args, **kwargs):
            # Create batch key based on function
            batch_key = func.__name__

            async with lock:
                if batch_key not in pending:
                    pending[batch_key] = []

                future = asyncio.Future()
                pending[batch_key].append((args, future))

                if len(pending[batch_key]) >= batch_size:
                    # Process batch immediately
                    batch = pending[batch_key]
                    pending[batch_key] = []

                    # Execute all in batch
                    for batch_args, batch_future in batch:
                        try:
                            result = await func(*batch_args, **kwargs)
                            batch_future.set_result(result)
                        except Exception as e:
                            batch_future.set_exception(e)

            # Wait for delay or batch fill
            await asyncio.sleep(delay_ms / 1000)

            return await future

        return batched

    def get_level(self) -> OptimizationLevel:
        return OptimizationLevel.MODERATE


class AdaptiveJIT:
    """Main self-adapting JIT optimizer"""

    def __init__(
        self,
        auto_optimize: bool = True,
        optimization_threshold: int = 50,
        cache_size: int = 10000,
    ):
        self.auto_optimize = auto_optimize
        self.optimization_threshold = optimization_threshold

        self.cache = AdaptiveCache(max_size=cache_size)
        self.hot_path_detector = HotPathDetector()
        self._strategies: Dict[OptimizationLevel, OptimizationStrategy] = {
            OptimizationLevel.BASIC: MemoizationStrategy(self.cache),
        }
        self._optimized_functions: Dict[str, Callable] = {}
        self._original_functions: Dict[str, Callable] = weakref.WeakValueDictionary()
        self._running = False
        self._optimizer_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the adaptive optimizer"""
        self._running = True
        if self.auto_optimize:
            self._optimizer_task = asyncio.create_task(self._optimization_loop())
        logger.info("Adaptive JIT started")

    async def stop(self):
        """Stop the adaptive optimizer"""
        self._running = False
        if self._optimizer_task:
            self._optimizer_task.cancel()
            try:
                await self._optimizer_task
            except asyncio.CancelledError:
                pass
        logger.info("Adaptive JIT stopped")

    async def _optimization_loop(self):
        """Background optimization loop"""
        while self._running:
            try:
                # Check for hot paths
                hot_paths = self.hot_path_detector.get_hot_paths()

                for func_id in hot_paths:
                    profile = self.hot_path_detector.get_profile(func_id)
                    if profile and profile.optimization_level == OptimizationLevel.NONE:
                        await self._apply_optimization(func_id, profile)

                await asyncio.sleep(5.0)
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(10.0)

    async def _apply_optimization(self, func_id: str, profile: ExecutionProfile):
        """Apply optimization to a function"""
        if func_id not in self._original_functions:
            return

        func = self._original_functions[func_id]

        # Select optimization level based on profile
        if profile.call_count > 1000 and profile.avg_time_ms > 10:
            level = OptimizationLevel.AGGRESSIVE
        elif profile.call_count > 500:
            level = OptimizationLevel.MODERATE
        else:
            level = OptimizationLevel.BASIC

        strategy = self._strategies.get(
            level, self._strategies[OptimizationLevel.BASIC]
        )

        optimized = strategy.optimize(func, profile)
        self._optimized_functions[func_id] = optimized
        profile.optimization_level = level

        logger.info(f"Applied {level.name} optimization to {func_id}")

    def profile(self, func: Callable = None, name: str = None):
        """Decorator to profile and optimize a function"""

        def decorator(fn: Callable) -> Callable:
            func_id = name or f"{fn.__module__}.{fn.__name__}"
            self._original_functions[func_id] = fn

            if asyncio.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def profiled_async(*args, **kwargs):
                    # Check for optimized version
                    if func_id in self._optimized_functions:
                        fn_to_call = self._optimized_functions[func_id]
                    else:
                        fn_to_call = fn

                    start = time.perf_counter()
                    try:
                        return await fn_to_call(*args, **kwargs)
                    finally:
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        self.hot_path_detector.record(func_id, elapsed_ms)

                return profiled_async
            else:

                @functools.wraps(fn)
                def profiled(*args, **kwargs):
                    if func_id in self._optimized_functions:
                        fn_to_call = self._optimized_functions[func_id]
                    else:
                        fn_to_call = fn

                    start = time.perf_counter()
                    try:
                        return fn_to_call(*args, **kwargs)
                    finally:
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        self.hot_path_detector.record(func_id, elapsed_ms)

                return profiled

        if func is not None:
            return decorator(func)
        return decorator

    def memoize(self, func: Callable = None, ttl: float = None):
        """Decorator for explicit memoization"""

        def decorator(fn: Callable) -> Callable:
            cache = AdaptiveCache(ttl_seconds=ttl or 300.0)

            if asyncio.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def memoized(*args, **kwargs):
                    key = hashlib.md5(
                        f"{fn.__name__}{args}{sorted(kwargs.items())}".encode()
                    ).hexdigest()

                    result = cache.get(key)
                    if result is not None:
                        return result

                    result = await fn(*args, **kwargs)
                    cache.put(key, result)
                    return result

                return memoized
            else:

                @functools.wraps(fn)
                def memoized(*args, **kwargs):
                    key = hashlib.md5(
                        f"{fn.__name__}{args}{sorted(kwargs.items())}".encode()
                    ).hexdigest()

                    result = cache.get(key)
                    if result is not None:
                        return result

                    result = fn(*args, **kwargs)
                    cache.put(key, result)
                    return result

                return memoized

        if func is not None:
            return decorator(func)
        return decorator

    def get_stats(self) -> Dict[str, Any]:
        """Get JIT statistics"""
        profiles = self.hot_path_detector.get_all_profiles()
        hot_paths = self.hot_path_detector.get_hot_paths()

        return {
            "totalFunctions": len(profiles),
            "hotPaths": len(hot_paths),
            "optimizedFunctions": len(self._optimized_functions),
            "cacheStats": self.cache.get_stats(),
            "topFunctions": sorted(
                [p.to_dict() for p in profiles],
                key=lambda x: x["callCount"],
                reverse=True,
            )[:10],
        }

    def get_profile(self, func_id: str) -> Optional[Dict[str, Any]]:
        """Get profile for a specific function"""
        profile = self.hot_path_detector.get_profile(func_id)
        return profile.to_dict() if profile else None


# Global JIT instance
_jit: Optional[AdaptiveJIT] = None


def get_adaptive_jit() -> AdaptiveJIT:
    """Get or create the adaptive JIT singleton"""
    global _jit
    if _jit is None:
        _jit = AdaptiveJIT()
    return _jit


# Convenience decorators
def jit_profile(func: Callable = None, name: str = None):
    """Decorator for JIT profiling"""
    return get_adaptive_jit().profile(func, name)


def jit_memoize(func: Callable = None, ttl: float = None):
    """Decorator for JIT memoization"""
    return get_adaptive_jit().memoize(func, ttl)
