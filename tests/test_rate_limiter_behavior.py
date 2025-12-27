import asyncio
import time

from core.rate_limiter import RateLimitManager, TokenBucketRateLimiter


def test_check_does_not_consume_tokens() -> None:
    async def _test() -> None:
        limiter = TokenBucketRateLimiter(max_requests=2, window_seconds=60)

        first_check = await limiter.check("user1")
        assert first_check.allowed
        assert first_check.remaining == 2

        after_check = await limiter.check("user1")
        assert after_check.allowed
        assert after_check.remaining == 2

        consumed = await limiter.consume("user1")
        assert consumed.allowed
        assert consumed.remaining == 1

    asyncio.run(_test())


def test_get_stats_prunes_expired_entries() -> None:
    limiter = TokenBucketRateLimiter(max_requests=3, window_seconds=1)

    async def _seed() -> None:
        await limiter.consume("user1")
        await limiter.consume("user1")

    asyncio.run(_seed())

    bucket = limiter.buckets["user1"]
    bucket[0] = time.time() - 5
    bucket[1] = time.time() - 4

    stats = limiter.get_stats("user1")
    assert stats["requests"] == 0
    assert stats["remaining"] == 3


def test_manager_check_does_not_consume() -> None:
    async def _test() -> None:
        manager = RateLimitManager()
        manager.add_limiter("api", 1, 60)

        allowed = await manager.check("api", "user1")
        assert allowed

        second_allowed = await manager.check("api", "user1")
        assert second_allowed

        consumed = await manager.consume("api", "user1")
        assert consumed.allowed
        assert consumed.remaining == 0

    asyncio.run(_test())
