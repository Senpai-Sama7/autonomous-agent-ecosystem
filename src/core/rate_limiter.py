"""Rate limiting system built on a token bucket algorithm."""
import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional


@dataclass
class RateLimitInfo:
    allowed: bool
    remaining: int
    reset_at: float
    retry_after: float = 0.0


class TokenBucketRateLimiter:
    """Token bucket rate limiter with sliding window semantics."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.buckets: Dict[str, Deque[float]] = {}
        self._lock = asyncio.Lock()

    def _get_bucket(self, key: str) -> Deque[float]:
        return self.buckets.setdefault(key, deque())

    def _prune_bucket(self, bucket: Deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

    def _build_info(self, bucket: Deque[float], now: float, allowed: bool, remaining: int) -> RateLimitInfo:
        reset_at = bucket[0] + self.window_seconds if bucket else now + self.window_seconds
        retry_after = max(0.0, reset_at - now) if not allowed else 0.0
        return RateLimitInfo(allowed, max(0, remaining), reset_at, retry_after)

    async def check(self, key: str) -> RateLimitInfo:
        """Check rate limit status without consuming a token."""
        async with self._lock:
            now = time.time()
            bucket = self._get_bucket(key)
            self._prune_bucket(bucket, now)

            remaining = self.max_requests - len(bucket)
            allowed = remaining > 0

            return self._build_info(bucket, now, allowed, remaining)

    async def consume(self, key: str) -> RateLimitInfo:
        """Check and consume a token if allowed."""
        async with self._lock:
            now = time.time()
            bucket = self._get_bucket(key)
            self._prune_bucket(bucket, now)

            remaining = self.max_requests - len(bucket)
            allowed = remaining > 0

            if allowed:
                bucket.append(now)
                remaining -= 1

            return self._build_info(bucket, now, allowed, remaining)

    def get_stats(self, key: str) -> Dict[str, int | float]:
        """Get current stats for key, pruning expired entries."""
        if key not in self.buckets:
            return {"requests": 0, "remaining": self.max_requests, "max": self.max_requests, "window": self.window_seconds}

        now = time.time()
        bucket = self._get_bucket(key)
        self._prune_bucket(bucket, now)

        return {
            "requests": len(bucket),
            "remaining": max(0, self.max_requests - len(bucket)),
            "max": self.max_requests,
            "window": self.window_seconds,
        }

    def clear(self, key: Optional[str] = None) -> None:
        """Clear rate limit data."""
        if key:
            self.buckets.pop(key, None)
        else:
            self.buckets.clear()


class RateLimitManager:
    """Manage multiple rate limiters for different resources."""

    def __init__(self) -> None:
        self.limiters: Dict[str, TokenBucketRateLimiter] = {}

    def add_limiter(self, name: str, max_requests: int, window_seconds: int) -> None:
        self.limiters[name] = TokenBucketRateLimiter(max_requests, window_seconds)

    async def check(self, limiter_name: str, key: str) -> bool:
        if limiter_name not in self.limiters:
            return True
        info = await self.limiters[limiter_name].check(key)
        return info.allowed

    async def consume(self, limiter_name: str, key: str) -> RateLimitInfo:
        if limiter_name not in self.limiters:
            return RateLimitInfo(True, 999, time.time())
        return await self.limiters[limiter_name].consume(key)

    def get_stats(self, limiter_name: str, key: str) -> Dict[str, int | float]:
        if limiter_name not in self.limiters:
            return {}
        return self.limiters[limiter_name].get_stats(key)


_manager: Optional[RateLimitManager] = None


def get_rate_limit_manager() -> RateLimitManager:
    """Get or create rate limit manager with defaults."""
    global _manager
    if _manager is None:
        _manager = RateLimitManager()
        _manager.add_limiter("task_submission", 100, 60)
        _manager.add_limiter("workflow_creation", 20, 60)
        _manager.add_limiter("api_calls", 1000, 60)
    return _manager
