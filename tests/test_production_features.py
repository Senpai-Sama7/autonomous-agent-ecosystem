"""
Tests for production features: logging, health, rate limiting, metrics
"""
import asyncio
import json
import logging
import sys
import os
import time
from io import StringIO

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestStructuredLogging:
    """Test structured logging system."""
    
    def test_json_formatter(self):
        from utils.structured_logger import StructuredFormatter
        
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="Test message", args=(), exc_info=None
        )
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data["level"] == "INFO"
        assert data["msg"] == "Test message"
        assert "ts" in data
    
    def test_console_formatter(self):
        from utils.structured_logger import ConsoleFormatter
        
        formatter = ConsoleFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="Test message", args=(), exc_info=None
        )
        
        output = formatter.format(record)
        assert "INFO" in output
        assert "Test message" in output
    
    def test_log_context(self):
        from utils.structured_logger import LogContext, request_id_var
        
        assert request_id_var.get() == ''
        
        with LogContext(request_id="test-123"):
            assert request_id_var.get() == "test-123"
        
        assert request_id_var.get() == ''
    
    def test_get_logger(self):
        from utils.structured_logger import get_logger
        
        logger = get_logger("test_module")
        assert logger.name == "test_module"


class TestHealthChecks:
    """Test health check system."""
    
    def test_health_status_dataclass(self):
        from api.health import HealthStatus
        
        status = HealthStatus("healthy", "All good", 10.5, {"key": "value"})
        assert status.status == "healthy"
        assert status.message == "All good"
        assert status.response_time_ms == 10.5
    
    def test_health_checker_init(self):
        from api.health import HealthChecker
        
        checker = HealthChecker()
        assert checker.uptime >= 0
        assert not checker._initialized
    
    def test_health_checker_set_components(self):
        from api.health import HealthChecker
        
        checker = HealthChecker()
        checker.set_components(agent_engine="mock", db_manager="mock")
        assert checker._initialized
    
    def test_get_health_checker_singleton(self):
        from api.health import get_health_checker
        
        checker1 = get_health_checker()
        checker2 = get_health_checker()
        assert checker1 is checker2


class TestRateLimiting:
    """Test rate limiting system."""
    
    def test_token_bucket_allows_requests(self):
        async def _test():
            from core.rate_limiter import TokenBucketRateLimiter
            
            limiter = TokenBucketRateLimiter(max_requests=5, window_seconds=60)
            
            for i in range(5):
                info = await limiter.consume("test_key")
                assert info.allowed, f"Request {i+1} should be allowed"
        
        asyncio.run(_test())
    
    def test_token_bucket_blocks_excess(self):
        async def _test():
            from core.rate_limiter import TokenBucketRateLimiter
            
            limiter = TokenBucketRateLimiter(max_requests=3, window_seconds=60)
            
            for _ in range(3):
                await limiter.consume("test_key")
            
            info = await limiter.consume("test_key")
            assert not info.allowed
            assert info.retry_after > 0
        
        asyncio.run(_test())
    
    def test_rate_limit_manager(self):
        async def _test():
            from core.rate_limiter import RateLimitManager
            
            manager = RateLimitManager()
            manager.add_limiter("test", 10, 60)
            
            allowed = await manager.check("test", "user1")
            assert allowed
        
        asyncio.run(_test())
    
    def test_get_stats(self):
        async def _test():
            from core.rate_limiter import TokenBucketRateLimiter
            
            limiter = TokenBucketRateLimiter(max_requests=10, window_seconds=60)
            await limiter.consume("test_key")
            await limiter.consume("test_key")
            
            stats = limiter.get_stats("test_key")
            assert stats["requests"] == 2
            assert stats["remaining"] == 8
        
        asyncio.run(_test())


class TestMetrics:
    """Test Prometheus metrics system."""
    
    def test_metrics_collector_init(self):
        from monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector()
        assert collector.start_time > 0
    
    def test_record_task(self):
        from monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.record_task_start("agent_001")
        collector.record_task_complete("agent_001", 1.5, True)
        # Should not raise
    
    def test_record_workflow(self):
        from monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.record_workflow_start()
        collector.record_workflow_complete(10.0, "completed")
        # Should not raise
    
    def test_record_error(self):
        from monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.record_error("ValueError")
        # Should not raise
    
    def test_get_metrics(self):
        from monitoring.metrics import get_metrics
        
        metrics = get_metrics()
        assert isinstance(metrics, bytes)
        assert b"astro_" in metrics
    
    def test_update_agent_health(self):
        from monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.update_agent_health("agent_001", True)
        collector.update_agent_health("agent_002", False)
        # Should not raise


class TestLogPerformanceDecorator:
    """Test log_performance decorator."""
    
    def test_sync_function(self):
        from utils.structured_logger import log_performance
        
        @log_performance
        def sync_func():
            return 42
        
        result = sync_func()
        assert result == 42
    
    def test_async_function(self):
        from utils.structured_logger import log_performance
        
        @log_performance
        async def async_func():
            await asyncio.sleep(0.01)
            return 42
        
        result = asyncio.run(async_func())
        assert result == 42
    
    def test_exception_handling(self):
        from utils.structured_logger import log_performance
        
        @log_performance
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_func()
