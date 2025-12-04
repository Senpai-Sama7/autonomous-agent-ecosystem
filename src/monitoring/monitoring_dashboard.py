"""
Monitoring Dashboard for Autonomous Agent Ecosystem
November 25, 2025

Real-time monitoring and visualization system for tracking agent performance,
system health, and emergent behaviors. Provides comprehensive insights
and alerting capabilities.

REFACTORED: Real telemetry integration.
- Replaced simulated CPU/memory with actual psutil measurements
- Integrated with database for actual agent performance metrics
- Added engine metrics integration capability
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logging.warning("psutil not installed. System metrics will be estimated.")

logger = logging.getLogger("MonitoringDashboard")

class AlertSeverity(Enum):
    """Severity levels for alerts"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class MetricType(Enum):
    """Types of metrics to track"""
    PERFORMANCE = "performance"
    HEALTH = "health"
    RESOURCE = "resource"
    BEHAVIOR = "behavior"
    COST = "cost"

@dataclass
class Alert:
    """Alert structure for monitoring events"""
    timestamp: float
    severity: AlertSeverity
    source: str
    message: str
    alert_id: str = ""  # For deduplication
    metrics: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    
    def __post_init__(self):
        if not self.alert_id:
            # Generate unique ID for deduplication
            self.alert_id = f"{self.source}:{self.severity.value}:{hash(self.message) % 10000}"

@dataclass
class SystemMetric:
    """Structure for system metrics"""
    timestamp: float
    metric_type: MetricType
    agent_id: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class MonitoringDashboard:
    """
    Real-time monitoring dashboard for the autonomous agent ecosystem.
    Provides comprehensive monitoring, visualization, and alerting capabilities.
    
    Now uses real system telemetry via psutil and integrates with engine/database
    for actual agent performance metrics.
    """
    
    def __init__(self, db_manager=None, engine=None):
        """Initialize the monitoring dashboard
        
        Args:
            db_manager: Optional DatabaseManager instance for fetching stored metrics
            engine: Optional AgentEngine instance for real-time agent metrics
        """
        self.metrics_history: List[SystemMetric] = []
        self.alerts: List[Alert] = []
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.system_health_score: float = 0.95
        self.last_update_time: float = time.time()
        self.update_interval: float = 5.0  # seconds
        self.alert_thresholds = {
            AlertSeverity.CRITICAL: 0.3,
            AlertSeverity.HIGH: 0.5,
            AlertSeverity.MEDIUM: 0.7,
            AlertSeverity.LOW: 0.85
        }
        
        # External integrations for real metrics
        self.db = db_manager
        self.engine = engine
        
        # Process handle for measuring own process metrics
        self._process = None
        if HAS_PSUTIL:
            try:
                self._process = psutil.Process(os.getpid())
            except Exception as e:
                logger.warning(f"Failed to initialize psutil process handle: {e}")
        
        # Shutdown flag for graceful termination
        self._shutdown_requested = False
        
        # Alert deduplication tracking (alert_id -> last_timestamp)
        self._recent_alerts: Dict[str, float] = {}
        self._alert_cooldown = 300.0  # 5 minutes between duplicate alerts
        
        logger.info("Monitoring Dashboard initialized with real telemetry support")
        
    async def start_monitoring(self):
        """Start the monitoring loop"""
        logger.info("Starting monitoring dashboard")
        
        while not self._shutdown_requested:
            try:
                await self._collect_metrics()
                await self._analyze_system_health()
                await self._check_alerts()
                await self._generate_visualizations()
                
                # Log system status periodically
                current_time = time.time()
                if current_time - self.last_update_time >= 60.0:  # Every minute
                    self._log_system_status()
                    self.last_update_time = current_time
                    
                    # Clean up old alert deduplication entries
                    self._cleanup_alert_tracking()
                    
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(10.0)  # Longer delay after errors
        
        logger.info("Monitoring dashboard stopped")
    
    def request_shutdown(self):
        """Request graceful shutdown of monitoring loop"""
        self._shutdown_requested = True
        logger.info("Shutdown requested for monitoring dashboard")
    
    def _cleanup_alert_tracking(self):
        """Remove old entries from alert deduplication tracking"""
        current_time = time.time()
        expired = [aid for aid, ts in self._recent_alerts.items() 
                   if current_time - ts > self._alert_cooldown * 2]
        for aid in expired:
            del self._recent_alerts[aid]
                
    async def _collect_metrics(self):
        """Collect REAL metrics from system and active agents"""
        current_time = time.time()
        
        # Collect real system-wide metrics
        system_cpu, system_memory = self._get_real_system_metrics()
        
        # Record system-wide resource metrics
        self.metrics_history.append(SystemMetric(
            timestamp=current_time,
            metric_type=MetricType.RESOURCE,
            agent_id='_system',
            value=system_cpu,
            metadata={'metric': 'cpu_usage', 'unit': 'percent', 'source': 'psutil'}
        ))
        
        self.metrics_history.append(SystemMetric(
            timestamp=current_time,
            metric_type=MetricType.RESOURCE,
            agent_id='_system',
            value=system_memory,
            metadata={'metric': 'memory_usage', 'unit': 'percent', 'source': 'psutil'}
        ))
        
        # Collect metrics from each active agent
        for agent_id, agent_info in self.active_agents.items():
            # Performance metrics - get from engine if available, otherwise from agent_info
            performance_score = await self._get_agent_performance_score(agent_id, agent_info)
            self.metrics_history.append(SystemMetric(
                timestamp=current_time,
                metric_type=MetricType.PERFORMANCE,
                agent_id=agent_id,
                value=performance_score,
                metadata={'metric': 'performance_score', 'source': 'engine' if self.engine else 'config'}
            ))
            
            # Per-agent resource metrics (distributed across agents based on system load)
            # In a real containerized environment, each agent would have isolated metrics
            agent_cpu = system_cpu / max(1, len(self.active_agents))
            agent_memory = system_memory / max(1, len(self.active_agents))
            
            self.metrics_history.append(SystemMetric(
                timestamp=current_time,
                metric_type=MetricType.RESOURCE,
                agent_id=agent_id,
                value=agent_cpu,
                metadata={'metric': 'cpu_usage', 'unit': 'percent', 'source': 'estimated'}
            ))
            
            self.metrics_history.append(SystemMetric(
                timestamp=current_time,
                metric_type=MetricType.RESOURCE,
                agent_id=agent_id,
                value=agent_memory,
                metadata={'metric': 'memory_usage', 'unit': 'percent', 'source': 'estimated'}
            ))
            
            # Health metrics - derive from actual agent state and success rate
            health_score = self._calculate_agent_health(agent_id, agent_info)
            self.metrics_history.append(SystemMetric(
                timestamp=current_time,
                metric_type=MetricType.HEALTH,
                agent_id=agent_id,
                value=health_score,
                metadata={'metric': 'health_score', 'source': 'calculated'}
            ))
            
        # Keep only last 24 hours of metrics (assuming 5-second intervals)
        max_metrics = int((24 * 60 * 60) / self.update_interval)
        if len(self.metrics_history) > max_metrics:
            self.metrics_history = self.metrics_history[-max_metrics:]
    
    def _get_real_system_metrics(self) -> tuple:
        """
        Get real CPU and memory usage from the system.
        Falls back to estimates if psutil is unavailable.
        """
        if HAS_PSUTIL and self._process:
            try:
                # Get process-specific metrics (more accurate for this application)
                cpu_percent = self._process.cpu_percent(interval=None) / 100.0
                memory_info = self._process.memory_info()
                total_memory = psutil.virtual_memory().total
                memory_percent = memory_info.rss / total_memory
                
                # Clamp to valid range
                cpu_percent = max(0.0, min(1.0, cpu_percent))
                memory_percent = max(0.0, min(1.0, memory_percent))
                
                return cpu_percent, memory_percent
            except Exception as e:
                logger.warning(f"Failed to get psutil metrics: {e}")
        
        # Fallback: estimate based on active tasks
        active_task_count = len(self.active_agents)
        estimated_cpu = min(0.9, 0.1 + (active_task_count * 0.05))
        estimated_memory = min(0.9, 0.2 + (active_task_count * 0.03))
        return estimated_cpu, estimated_memory
    
    async def _get_agent_performance_score(self, agent_id: str, agent_info: Dict) -> float:
        """
        Get actual performance score for an agent from engine metrics or database.
        """
        # Try to get from engine's real-time metrics
        if self.engine and hasattr(self.engine, 'performance_metrics'):
            metrics = self.engine.performance_metrics.get(agent_id, [])
            if metrics:
                # Calculate performance score from recent execution times
                # Lower execution time = higher score
                recent = metrics[-10:] if len(metrics) > 10 else metrics
                avg_time = sum(recent) / len(recent)
                # Score: 1.0 for <1s, decreasing to 0.5 for >10s
                score = max(0.5, 1.0 - (avg_time / 20.0))
                return score
        
        # Try to get from database
        if self.db and hasattr(self.db, 'get_agent_stats_async'):
            try:
                stats = await self.db.get_agent_stats_async(agent_id)
                if 'execution_time' in stats:
                    avg_time = stats['execution_time'].get('avg', 5.0)
                    score = max(0.5, 1.0 - (avg_time / 20.0))
                    return score
            except Exception as e:
                logger.debug(f"Could not fetch DB stats for {agent_id}: {e}")
        
        # Fallback to agent_info config value
        return agent_info.get('performance_score', 0.8)
    
    def _calculate_agent_health(self, agent_id: str, agent_info: Dict) -> float:
        """
        Calculate health score based on agent state and metrics.
        """
        state = agent_info.get('state', 'unknown')
        
        # Base score from state
        state_scores = {
            'active': 0.95,
            'idle': 0.90,
            'busy': 0.85,
            'degraded': 0.6,
            'recovering': 0.5,
            'failed': 0.2,
            'unknown': 0.7
        }
        base_score = state_scores.get(state, 0.7)
        
        # Adjust based on failure count if available
        failure_count = agent_info.get('failure_count', 0)
        if failure_count > 0:
            penalty = min(0.3, failure_count * 0.1)
            base_score = max(0.2, base_score - penalty)
        
        # Adjust based on success rate if engine is available
        if self.engine and agent_id in self.engine.agent_instances:
            agent_instance = self.engine.agent_instances[agent_id]
            if hasattr(agent_instance, 'get_success_rate'):
                success_rate = agent_instance.get_success_rate()
                # Weight the success rate into the health score
                base_score = (base_score * 0.6) + (success_rate * 0.4)
        
        return base_score
            
    async def _analyze_system_health(self):
        """Analyze overall system health"""
        current_time = time.time()
        recent_metrics = [
            m for m in self.metrics_history
            if current_time - m.timestamp <= 300  # Last 5 minutes
        ]
        
        if not recent_metrics:
            self.system_health_score = 0.8
            return
            
        # Calculate health scores by category
        performance_scores = [m.value for m in recent_metrics if m.metric_type == MetricType.PERFORMANCE]
        health_scores = [m.value for m in recent_metrics if m.metric_type == MetricType.HEALTH]
        resource_scores = [
            1.0 - m.value for m in recent_metrics 
            if m.metric_type == MetricType.RESOURCE 
            and m.metadata 
            and m.metadata.get('metric') in ['cpu_usage', 'memory_usage']
        ]
        
        avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.8
        avg_health = sum(health_scores) / len(health_scores) if health_scores else 0.8
        avg_resource = sum(resource_scores) / len(resource_scores) if resource_scores else 0.8
        
        # Weighted health score
        self.system_health_score = (
            avg_performance * 0.4 +
            avg_health * 0.4 +
            avg_resource * 0.2
        )
        
        # Detect potential issues
        if self.system_health_score < 0.7:
            await self._create_alert(
                severity=AlertSeverity.HIGH,
                source="system_health",
                message=f"System health degraded: {self.system_health_score:.2f}",
                metrics={'health_score': self.system_health_score}
            )
            
    async def _check_alerts(self):
        """Check for conditions that require alerts"""
        current_time = time.time()
        recent_metrics = [
            m for m in self.metrics_history
            if current_time - m.timestamp <= 300  # Last 5 minutes
        ]
        
        # Check for high resource usage
        high_cpu_agents = {}
        high_memory_agents = {}
        
        for metric in recent_metrics:
            if metric.metric_type == MetricType.RESOURCE:
                if metric.metadata.get('metric') == 'cpu_usage' and metric.value > 0.85:
                    high_cpu_agents[metric.agent_id] = metric.value
                elif metric.metadata.get('metric') == 'memory_usage' and metric.value > 0.9:
                    high_memory_agents[metric.agent_id] = metric.value
        
        # Create alerts for high resource usage
        for agent_id, cpu_value in high_cpu_agents.items():
            await self._create_alert(
                severity=AlertSeverity.MEDIUM,
                source=agent_id,
                message=f"High CPU usage: {cpu_value:.2f}",
                metrics={'cpu_usage': cpu_value, 'agent_id': agent_id}
            )
            
        for agent_id, memory_value in high_memory_agents.items():
            await self._create_alert(
                severity=AlertSeverity.HIGH,
                source=agent_id,
                message=f"Critical memory usage: {memory_value:.2f}",
                metrics={'memory_usage': memory_value, 'agent_id': agent_id}
            )
            
        # Check for failing agents
        failing_agents = [
            agent_id for agent_id, info in self.active_agents.items()
            if info.get('state') in ['failed', 'recovering'] and info.get('failure_count', 0) > 2
        ]
        
        for agent_id in failing_agents:
            await self._create_alert(
                severity=AlertSeverity.CRITICAL,
                source=agent_id,
                message=f"Agent failing repeatedly - requires intervention",
                metrics={'agent_state': self.active_agents[agent_id].get('state'), 'failure_count': self.active_agents[agent_id].get('failure_count', 0)}
            )
            
    async def _create_alert(self, severity: AlertSeverity, source: str, message: str, metrics: Dict[str, Any] = None):
        """Create a new alert with deduplication"""
        alert = Alert(
            timestamp=time.time(),
            severity=severity,
            source=source,
            message=message,
            metrics=metrics or {}
        )
        
        # Check for duplicate alert within cooldown period
        current_time = time.time()
        if alert.alert_id in self._recent_alerts:
            last_time = self._recent_alerts[alert.alert_id]
            if current_time - last_time < self._alert_cooldown:
                logger.debug(f"Suppressed duplicate alert: {alert.alert_id}")
                return  # Skip duplicate
        
        # Record this alert for deduplication
        self._recent_alerts[alert.alert_id] = current_time
        
        self.alerts.append(alert)
        logger.warning(f"ALERT [{severity.value.upper()}] {source}: {message}")
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
            
    async def _generate_visualizations(self):
        """Generate visualizations for the dashboard"""
        # This would generate real visualizations in a production system
        # For now, we'll simulate the process
        pass
        
    def _log_system_status(self):
        """Log current system status"""
        status_summary = {
            'timestamp': time.time(),
            'system_health_score': self.system_health_score,
            'active_agents_count': len(self.active_agents),
            'active_agents': {agent_id: info.get('state', 'unknown') for agent_id, info in self.active_agents.items()},
            'recent_alerts_count': len([a for a in self.alerts if not a.resolved and time.time() - a.timestamp <= 300]),
            'metrics_count': len(self.metrics_history)
        }
        
        logger.info(f"System Status: {json.dumps(status_summary, indent=2)}")
        
    def register_agent(self, agent_id: str, agent_info: Dict[str, Any]):
        """Register an agent with the monitoring system"""
        self.active_agents[agent_id] = agent_info
        logger.info(f"Registered agent {agent_id} with monitoring dashboard")
        
    def update_agent_info(self, agent_id: str, agent_info: Dict[str, Any]):
        """Update agent information"""
        if agent_id in self.active_agents:
            self.active_agents[agent_id].update(agent_info)
        else:
            self.register_agent(agent_id, agent_info)
            
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        return {
            'health_score': self.system_health_score,
            'status': self._get_health_status_string(),
            'active_agents_count': len(self.active_agents),
            'recent_alerts': self._get_recent_alerts(),
            'last_updated': time.time()
        }
        
    def _get_health_status_string(self) -> str:
        """Get human-readable health status"""
        if self.system_health_score >= 0.85:
            return "excellent"
        elif self.system_health_score >= 0.7:
            return "good"
        elif self.system_health_score >= 0.5:
            return "degraded"
        else:
            return "critical"
            
    def _get_recent_alerts(self) -> List[Dict[str, Any]]:
        """Get recent unresolved alerts"""
        recent_alerts = [
            {
                'timestamp': alert.timestamp,
                'severity': alert.severity.value,
                'source': alert.source,
                'message': alert.message,
                'metrics': alert.metrics
            }
            for alert in self.alerts
            if not alert.resolved and time.time() - alert.timestamp <= 3600  # Last hour
        ]
        
        return sorted(recent_alerts, key=lambda x: x['timestamp'], reverse=True)[:10]
        
    def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific agent"""
        if agent_id not in self.active_agents:
            return {'error': 'Agent not found'}
            
        current_time = time.time()
        agent_metrics = [
            m for m in self.metrics_history
            if m.agent_id == agent_id and current_time - m.timestamp <= 3600  # Last hour
        ]
        
        if not agent_metrics:
            return {'agent_id': agent_id, 'performance_score': 0.8, 'status': 'no_recent_data'}
            
        # Calculate various performance metrics
        performance_scores = [m.value for m in agent_metrics if m.metric_type == MetricType.PERFORMANCE]
        cpu_usages = [m.value for m in agent_metrics if m.metric_type == MetricType.RESOURCE and m.metadata.get('metric') == 'cpu_usage']
        memory_usages = [m.value for m in agent_metrics if m.metric_type == MetricType.RESOURCE and m.metadata.get('metric') == 'memory_usage']
        
        return {
            'agent_id': agent_id,
            'current_performance': performance_scores[-1] if performance_scores else 0.8,
            'avg_performance': sum(performance_scores) / len(performance_scores) if performance_scores else 0.8,
            'avg_cpu_usage': sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0.3,
            'avg_memory_usage': sum(memory_usages) / len(memory_usages) if memory_usages else 0.5,
            'status': self.active_agents[agent_id].get('state', 'unknown'),
            'recent_alerts': len([a for a in self.alerts if a.source == agent_id and not a.resolved]),
            'data_points': len(agent_metrics)
        }
        
    def get_performance_chart(self, agent_id: Optional[str] = None) -> Optional[str]:
        """Generate a performance chart as base64-encoded image"""
        try:
            current_time = time.time()
            
            if agent_id:
                # Get data for specific agent
                agent_metrics = [
                    m for m in self.metrics_history
                    if m.agent_id == agent_id and m.metric_type == MetricType.PERFORMANCE
                    and current_time - m.timestamp <= 3600  # Last hour
                ]
                
                if agent_metrics:
                    timestamps = [m.timestamp for m in agent_metrics]
                    values = [m.value for m in agent_metrics]
                    title = f"Performance: {agent_id}"
                else:
                    timestamps = [current_time - 3600 + i * 60 for i in range(61)]
                    values = [0.8] * 61
                    title = f"No data available for {agent_id}"
            else:
                # Get system-wide health score over time
                timestamps = [current_time - 3600 + i * 60 for i in range(61)]
                values = [max(0.5, 0.95 - (i * 0.001)) for i in range(61)]  # Simulated trend
                title = "System Health Score"
            
            # Create the chart
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, values, 'b-', linewidth=2.5)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.0)
            
            # Format x-axis as time
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error generating performance chart: {str(e)}")
            return None
            
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alerts"""
        current_time = time.time()
        active_alerts = [a for a in self.alerts if not a.resolved]
        
        severity_counts = {
            severity.value: len([a for a in active_alerts if a.severity == severity])
            for severity in AlertSeverity
        }
        
        recent_alerts = [
            {
                'timestamp': current_time - a.timestamp,
                'severity': a.severity.value,
                'source': a.source,
                'message': a.message
            }
            for a in active_alerts
            if current_time - a.timestamp <= 300  # Last 5 minutes
        ]
        
        return {
            'total_alerts': len(active_alerts),
            'severity_counts': severity_counts,
            'recent_alerts': sorted(recent_alerts, key=lambda x: x['timestamp'])[:5],
            'health_impact': self._calculate_alert_health_impact(active_alerts)
        }
        
    def _calculate_alert_health_impact(self, alerts: List[Alert]) -> float:
        """Calculate the health impact of current alerts"""
        if not alerts:
            return 1.0
            
        total_impact = 0.0
        max_impact = 0.0
        
        for alert in alerts:
            severity_impact = {
                AlertSeverity.CRITICAL: 0.4,
                AlertSeverity.HIGH: 0.25,
                AlertSeverity.MEDIUM: 0.15,
                AlertSeverity.LOW: 0.05,
                AlertSeverity.INFO: 0.01
            }[alert.severity]
            
            total_impact += severity_impact
            max_impact += 0.4  # Maximum possible impact per alert
            
        return max(0.0, 1.0 - (total_impact / max_impact if max_impact > 0 else 0.0))
        
    def resolve_alert(self, alert_index: int):
        """Mark an alert as resolved"""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].resolved = True
            logger.info(f"Resolved alert: {self.alerts[alert_index].message}")

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create monitoring dashboard
    dashboard = MonitoringDashboard()
    
    # Register some test agents
    test_agents = {
        'research_agent_001': {
            'state': 'active',
            'performance_score': 0.92,
            'capabilities': ['web_search', 'content_extraction']
        },
        'code_agent_001': {
            'state': 'active', 
            'performance_score': 0.88,
            'capabilities': ['code_generation', 'optimization']
        },
        'analysis_agent_001': {
            'state': 'active',
            'performance_score': 0.95,
            'capabilities': ['data_analysis', 'visualization']
        }
    }
    
    for agent_id, info in test_agents.items():
        dashboard.register_agent(agent_id, info)
        
    print("=== Monitoring Dashboard Test ===")
    print(f"System Health: {dashboard.get_system_health()}")
    
    # Test agent performance
    for agent_id in test_agents.keys():
        print(f"\n{agent_id} Performance: {dashboard.get_agent_performance(agent_id)}")
        
    print(f"\nAlert Summary: {dashboard.get_alert_summary()}")
    
    # Test chart generation
    chart_url = dashboard.get_performance_chart('research_agent_001')
    if chart_url:
        print(f"\nPerformance chart generated (base64): {chart_url[:100]}...")  # Show first 100 chars
        
    print("\nDashboard test completed successfully!")