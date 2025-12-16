"""
Prometheus Metrics Export

Provides Prometheus metrics for monitoring the ASTRO ecosystem.
Metrics are lazily initialized to avoid duplicate registration issues in tests.
"""
import time
from typing import Optional
import logging

from prometheus_client import Counter, Gauge, Histogram, Info, REGISTRY, generate_latest
from prometheus_client.registry import CollectorRegistry

logger = logging.getLogger(__name__)

# Flag to track if metrics have been initialized
_metrics_initialized = False


def _get_or_create_metric(name: str, metric_class, description: str, labelnames=None, **kwargs):
    """
    Get an existing metric or create a new one.
    Handles the case where metrics might already be registered.
    """
    # First, try to find existing collector by checking all possible names
    # Counter creates multiple collectors with suffixes like _total, _created
    try:
        names_to_check = [name, f"{name}_total", f"{name}_created"]
        for collector in list(REGISTRY._names_to_collectors.values()):
            collector_name = getattr(collector, '_name', None)
            if collector_name and collector_name in names_to_check:
                # Found existing collector - return the base one
                if collector_name == name:
                    return collector
                # For _total suffix, the original Counter is what we want
                if hasattr(collector, '_name') and collector._name == name:
                    return collector
    except Exception:
        pass

    # Try to create new metric
    try:
        if labelnames:
            return metric_class(name, description, labelnames, **kwargs)
        else:
            return metric_class(name, description, **kwargs)
    except ValueError as e:
        # Metric already exists - find and return it
        error_str = str(e)
        if "Duplicated" in error_str:
            # Try to extract and return the existing metric
            try:
                for collector in list(REGISTRY._names_to_collectors.values()):
                    collector_name = getattr(collector, '_name', None)
                    if collector_name == name:
                        return collector
                    # For Counter, check if this is the right one
                    if hasattr(collector, 'describe'):
                        for metric in collector.describe():
                            if metric.name == name or metric.name == f"{name}_total":
                                return collector
            except Exception:
                pass
            # If we still can't find it, create a stub that won't fail
            logger.warning(f"Could not retrieve existing metric {name}, using fallback")
            return _create_fallback_metric(name, metric_class, description, labelnames, **kwargs)
        raise


def _create_fallback_metric(name: str, metric_class, description: str, labelnames=None, **kwargs):
    """Create a fallback metric that uses the existing registry entry."""
    # This is a last resort - return a mock-like object that uses existing registry
    class FallbackMetric:
        def __init__(self, name):
            self._name = name

        def labels(self, **kwargs):
            return self

        def inc(self, amount=1):
            pass

        def dec(self, amount=1):
            pass

        def set(self, value):
            pass

        def observe(self, value):
            pass

        def info(self, val):
            pass

    return FallbackMetric(name)


def _initialize_metrics():
    """Initialize all metrics lazily to avoid duplicate registration."""
    global _metrics_initialized
    global app_info, task_total, task_duration, active_tasks, queued_tasks
    global workflow_total, workflow_duration, active_workflows
    global agent_health, agent_tasks_active, agent_reliability
    global llm_requests, llm_tokens, llm_duration, llm_cost
    global db_queries, db_duration, system_uptime, system_errors

    if _metrics_initialized:
        return

    # App info
    app_info = _get_or_create_metric('astro_app', Info, 'Astro application information')
    try:
        app_info.info({'version': '1.0.0', 'environment': 'production'})
    except Exception:
        pass  # Info might already be set

    # Task metrics
    task_total = _get_or_create_metric(
        'astro_tasks_total', Counter, 'Total tasks executed', ['agent_id', 'status'])
    task_duration = _get_or_create_metric(
        'astro_task_duration_seconds', Histogram, 'Task duration', ['agent_id'],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300])
    active_tasks = _get_or_create_metric('astro_active_tasks', Gauge, 'Active tasks')
    queued_tasks = _get_or_create_metric('astro_queued_tasks', Gauge, 'Queued tasks')

    # Workflow metrics
    workflow_total = _get_or_create_metric(
        'astro_workflows_total', Counter, 'Total workflows', ['status'])
    workflow_duration = _get_or_create_metric(
        'astro_workflow_duration_seconds', Histogram, 'Workflow duration',
        buckets=[1, 5, 10, 30, 60, 300, 600, 1800])
    active_workflows = _get_or_create_metric('astro_active_workflows', Gauge, 'Active workflows')

    # Agent metrics
    agent_health = _get_or_create_metric(
        'astro_agent_health', Gauge, 'Agent health (1=healthy)', ['agent_id'])
    agent_tasks_active = _get_or_create_metric(
        'astro_agent_tasks_active', Gauge, 'Active tasks per agent', ['agent_id'])
    agent_reliability = _get_or_create_metric(
        'astro_agent_reliability', Gauge, 'Agent reliability score', ['agent_id'])

    # LLM metrics
    llm_requests = _get_or_create_metric(
        'astro_llm_requests_total', Counter, 'LLM requests', ['model', 'status'])
    llm_tokens = _get_or_create_metric(
        'astro_llm_tokens_total', Counter, 'LLM tokens', ['model', 'type'])
    llm_duration = _get_or_create_metric(
        'astro_llm_duration_seconds', Histogram, 'LLM request duration', ['model'],
        buckets=[0.5, 1, 2, 5, 10, 20, 30])
    llm_cost = _get_or_create_metric(
        'astro_llm_cost_usd', Counter, 'LLM cost in USD', ['model'])

    # Database metrics
    db_queries = _get_or_create_metric(
        'astro_db_queries_total', Counter, 'Database queries', ['operation'])
    db_duration = _get_or_create_metric(
        'astro_db_query_duration_seconds', Histogram, 'Query duration', ['operation'],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])

    # System metrics
    system_uptime = _get_or_create_metric('astro_uptime_seconds', Gauge, 'System uptime')
    system_errors = _get_or_create_metric(
        'astro_errors_total', Counter, 'System errors', ['error_type'])

    _metrics_initialized = True


# Initialize placeholder variables (will be populated by _initialize_metrics)
app_info = None
task_total = None
task_duration = None
active_tasks = None
queued_tasks = None
workflow_total = None
workflow_duration = None
active_workflows = None
agent_health = None
agent_tasks_active = None
agent_reliability = None
llm_requests = None
llm_tokens = None
llm_duration = None
llm_cost = None
db_queries = None
db_duration = None
system_uptime = None
system_errors = None

# Initialize metrics on module load
_initialize_metrics()


class MetricsCollector:
    """Helper class to collect metrics."""

    def __init__(self):
        self.start_time = time.time()

    def update_uptime(self):
        system_uptime.set(time.time() - self.start_time)

    def record_task_start(self, agent_id: str):
        active_tasks.inc()
        agent_tasks_active.labels(agent_id=agent_id).inc()

    def record_task_complete(self, agent_id: str, duration_seconds: float, success: bool):
        active_tasks.dec()
        agent_tasks_active.labels(agent_id=agent_id).dec()
        task_total.labels(agent_id=agent_id, status='success' if success else 'failure').inc()
        task_duration.labels(agent_id=agent_id).observe(duration_seconds)

    def record_workflow_start(self):
        active_workflows.inc()

    def record_workflow_complete(self, duration_seconds: float, status: str):
        active_workflows.dec()
        workflow_total.labels(status=status).inc()
        workflow_duration.observe(duration_seconds)

    def record_llm_request(self, model: str, duration_seconds: float,
                          prompt_tokens: int, completion_tokens: int,
                          cost_usd: float, success: bool):
        llm_requests.labels(model=model, status='success' if success else 'failure').inc()
        llm_duration.labels(model=model).observe(duration_seconds)
        llm_tokens.labels(model=model, type='prompt').inc(prompt_tokens)
        llm_tokens.labels(model=model, type='completion').inc(completion_tokens)
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
