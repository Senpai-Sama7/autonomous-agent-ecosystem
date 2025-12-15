"""
Prometheus Metrics Export
"""

import time
from typing import Any, Callable, Optional, Type, TypeVar

from prometheus_client import Counter, Gauge, Histogram, Info, REGISTRY, generate_latest

MetricT = TypeVar("MetricT")


def _get_existing_collector(name: str) -> Optional[Any]:
    """Return an already-registered collector if one exists."""
    # get_collector_for_name was added in 0.14.0
    if hasattr(REGISTRY, "get_collector_for_name"):
        return REGISTRY.get_collector_for_name(name)

    # Fallback for older versions
    names_to_collectors = getattr(REGISTRY, "_names_to_collectors", {})
    return names_to_collectors.get(name)


def _metric_or_existing(name: str, factory: Callable[[], MetricT], expected_type: Type[MetricT]) -> MetricT:
    """Reuse collectors already registered to the global registry to avoid duplicates."""
    existing = _get_existing_collector(name)
    if existing is not None:
        if not isinstance(existing, expected_type):
            raise TypeError(
                f"Collector '{name}' is already registered with incompatible type {type(existing)}"
            )
        return existing
    return factory()


# App info
app_info = Info("astro_app", "Astro application information")
app_info.info({"version": "1.0.0", "environment": "production"})

# Task metrics
task_total = Counter(
    "astro_tasks_total", "Total tasks executed", ["agent_id", "status"]
)
task_duration = Histogram(
    "astro_task_duration_seconds",
    "Task duration",
    ["agent_id"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
)
active_tasks = Gauge("astro_active_tasks", "Active tasks")
queued_tasks = Gauge("astro_queued_tasks", "Queued tasks")

# Workflow metrics
workflow_total = Counter("astro_workflows_total", "Total workflows", ["status"])
workflow_duration = Histogram(
    "astro_workflow_duration_seconds",
    "Workflow duration",
    buckets=[1, 5, 10, 30, 60, 300, 600, 1800],
)
active_workflows = Gauge("astro_active_workflows", "Active workflows")

# Agent metrics
agent_health = Gauge("astro_agent_health", "Agent health (1=healthy)", ["agent_id"])
agent_tasks_active = Gauge(
    "astro_agent_tasks_active", "Active tasks per agent", ["agent_id"]
)
agent_reliability = Gauge(
    "astro_agent_reliability", "Agent reliability score", ["agent_id"]
)

# LLM metrics
llm_requests = Counter("astro_llm_requests_total", "LLM requests", ["model", "status"])
llm_tokens = Counter("astro_llm_tokens_total", "LLM tokens", ["model", "type"])
llm_duration = Histogram(
    "astro_llm_duration_seconds",
    "LLM request duration",
    ["model"],
    buckets=[0.5, 1, 2, 5, 10, 20, 30],
)
llm_cost = Counter("astro_llm_cost_usd", "LLM cost in USD", ["model"])

# Database metrics
db_queries = Counter("astro_db_queries_total", "Database queries", ["operation"])
db_duration = Histogram(
    "astro_db_query_duration_seconds",
    "Query duration",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
)

# System metrics
system_uptime = Gauge("astro_uptime_seconds", "System uptime")
system_errors = Counter("astro_errors_total", "System errors", ["error_type"])


class MetricsCollector:
    """Helper class to collect metrics."""

    def __init__(self):
        self.start_time = time.time()

    def update_uptime(self):
        system_uptime.set(time.time() - self.start_time)

    def record_task_start(self, agent_id: str):
        active_tasks.inc()
        agent_tasks_active.labels(agent_id=agent_id).inc()

    def record_task_complete(
        self, agent_id: str, duration_seconds: float, success: bool
    ):
        active_tasks.dec()
        agent_tasks_active.labels(agent_id=agent_id).dec()
        task_total.labels(
            agent_id=agent_id, status="success" if success else "failure"
        ).inc()
        task_duration.labels(agent_id=agent_id).observe(duration_seconds)

    def record_workflow_start(self):
        active_workflows.inc()

    def record_workflow_complete(self, duration_seconds: float, status: str):
        active_workflows.dec()
        workflow_total.labels(status=status).inc()
        workflow_duration.observe(duration_seconds)

    def record_llm_request(
        self,
        model: str,
        duration_seconds: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        success: bool,
    ):
        llm_requests.labels(
            model=model, status="success" if success else "failure"
        ).inc()
        llm_duration.labels(model=model).observe(duration_seconds)
        llm_tokens.labels(model=model, type="prompt").inc(prompt_tokens)
        llm_tokens.labels(model=model, type="completion").inc(completion_tokens)
        llm_cost.labels(model=model).inc(cost_usd)

    def record_db_query(self, operation: str, duration_seconds: float):
        db_queries.labels(operation=operation).inc()
        db_duration.labels(operation=operation).observe(duration_seconds)

    def update_agent_health(self, agent_id: str, is_healthy: bool):
        agent_health.labels(agent_id=agent_id).set(1 if is_healthy else 0)

    def update_agent_reliability(self, agent_id: str, score: float):
        agent_reliability.labels(agent_id=agent_id).set(score)

    def record_error(self, error_type: str):
        system_errors.labels(error_type=error_type).inc()


_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


def get_metrics() -> bytes:
    """Get Prometheus metrics in text format."""
    return generate_latest(REGISTRY)
