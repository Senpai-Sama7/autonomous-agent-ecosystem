"""
ASTRO API Server
FastAPI-based REST API and WebSocket server for the ASTRO web interface.

This module provides:
- RESTful endpoints for system control, agents, workflows, chat, files
- WebSocket endpoint for real-time updates
- Static file serving for the web interface
- Integration with the core AgentEngine
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import security middleware
from api.middleware import (
    add_security_middleware,
    get_allowed_origins,
    security_config,
    authenticate_websocket,
)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.engine import AgentEngine, Workflow, Task, WorkflowPriority, AgentConfig
from core.nl_interface import NaturalLanguageInterface
from core.llm_factory import LLMFactory
from core.database import DatabaseManager
# Lazy imports for agents to handle missing dependencies gracefully
ResearchAgent = None
CodeAgent = None
FileSystemAgent = None

def _import_agents():
    """Import agents lazily to handle missing dependencies."""
    global ResearchAgent, CodeAgent, FileSystemAgent
    
    try:
        from agents.research_agent import ResearchAgent as _ResearchAgent
        ResearchAgent = _ResearchAgent
    except ImportError as e:
        logging.warning(f"ResearchAgent unavailable: {e}")
    
    try:
        from agents.code_agent import CodeAgent as _CodeAgent
        CodeAgent = _CodeAgent
    except ImportError as e:
        logging.warning(f"CodeAgent unavailable: {e}")
    
    try:
        from agents.filesystem_agent import FileSystemAgent as _FileSystemAgent
        FileSystemAgent = _FileSystemAgent
    except ImportError as e:
        logging.warning(f"FileSystemAgent unavailable: {e}")

logger = logging.getLogger("ASTRO.API")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class CommandRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=10000)

class ChatMessageRequest(BaseModel):
    session_id: str = Field(default="default")
    message: str = Field(..., min_length=1, max_length=10000)

class WorkflowCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(default="", max_length=2000)
    priority: str = Field(default="medium")
    tasks: List[Dict[str, Any]] = Field(default_factory=list)

class FileCreateRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=500)
    content: str = Field(default="")

class KnowledgeItemRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    type: str = Field(default="document")
    tags: List[str] = Field(default_factory=list)

class SystemStatusResponse(BaseModel):
    status: str
    agents_count: int
    active_workflows: int
    uptime: float
    version: str = "1.0.0"

class AgentResponse(BaseModel):
    id: str
    name: str
    icon: str
    status: str
    description: str
    capabilities: List[str]

class TelemetryResponse(BaseModel):
    signalIntegrity: float
    bandwidth: float
    load: float
    latency: float
    timestamp: str

# WebSocket message validation models
class WSMessageType(str, Enum):
    """Valid WebSocket message types."""
    PING = "ping"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    COMMAND = "command"

class WSMessage(BaseModel):
    """Validated WebSocket message structure."""
    type: WSMessageType
    payload: Dict[str, Any] = Field(default_factory=dict, max_length=100)
    
    class Config:
        max_anystr_length = 10000  # Limit string lengths in payload

# ============================================================================
# CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_metadata[websocket] = {
            "client_id": client_id or str(uuid.uuid4()),
            "connected_at": datetime.now().isoformat(),
        }
        logger.info(f"WebSocket client connected: {self.connection_metadata[websocket]['client_id']}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        metadata = self.connection_metadata.pop(websocket, {})
        logger.info(f"WebSocket client disconnected: {metadata.get('client_id', 'unknown')}")
    
    async def broadcast(self, message_type: str, payload: Any):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return
        
        message = json.dumps({"type": message_type, "payload": payload})
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.add(connection)
        
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_to_client(self, websocket: WebSocket, message_type: str, payload: Any):
        """Send a message to a specific client."""
        try:
            message = json.dumps({"type": message_type, "payload": payload})
            await websocket.send_text(message)
        except Exception as e:
            logger.warning(f"Failed to send to client: {e}")

manager = ConnectionManager()

# ============================================================================
# APPLICATION STATE
# ============================================================================

class AppState:
    """Global application state container."""
    
    def __init__(self):
        self.engine: Optional[AgentEngine] = None
        self.nl_interface: Optional[NaturalLanguageInterface] = None
        self.llm_client: Optional[Any] = None
        self.agents: Dict[str, Any] = {}
        self.db: Optional[DatabaseManager] = None
        self.start_time: float = time.time()
        self.running: bool = False
        self.chat_sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.knowledge_items: List[Dict[str, Any]] = []
        self._background_tasks: List[asyncio.Task] = []

app_state = AppState()

# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    logger.info("Starting ASTRO API Server...")
    
    # Initialize database
    app_state.db = DatabaseManager()
    
    # Initialize LLM client (Ollama by default)
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "llama3.2")
    app_state.llm_client = LLMFactory.create_client(provider=provider)
    app_state.llm_model = model
    if app_state.llm_client:
        logger.info(f"LLM client initialized: {provider}/{model}")
    else:
        logger.warning("No LLM client available - chat will be limited")
    
    # Initialize engine
    app_state.engine = AgentEngine()
    
    # Initialize agents
    await initialize_agents()
    
    # Start background tasks
    app_state._background_tasks.append(
        asyncio.create_task(telemetry_broadcaster())
    )
    app_state._background_tasks.append(
        asyncio.create_task(status_monitor())
    )
    
    logger.info("ASTRO API Server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ASTRO API Server...")
    
    for task in app_state._background_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    if app_state.engine:
        await app_state.engine.shutdown()
    
    logger.info("ASTRO API Server shutdown complete")

async def initialize_agents():
    """Initialize and register all agents with the engine."""
    
    # Import agents lazily
    _import_agents()
    
    # Research Agent
    if ResearchAgent is not None:
        research_config = {
            "search_provider": "duckduckgo",
            "max_results": 10,
            "safe_search": True,
        }
        research_agent = ResearchAgent(
            agent_id="research_agent_001",
            config=research_config
        )
        await app_state.engine.register_agent(
            AgentConfig(
                agent_id="research_agent_001",
                capabilities=["web_search", "content_extraction", "summarization"],
                max_concurrent_tasks=3,
            ),
            instance=research_agent
        )
        app_state.agents["research_agent_001"] = research_agent
        logger.info("Research agent initialized")
    else:
        logger.warning("Research agent not available - missing dependencies")
    
    # Code Agent
    if CodeAgent is not None:
        try:
            code_config = {
                "safe_mode": True,
                "use_docker_sandbox": True,
                "allowed_languages": ["python"],
            }
            code_agent = CodeAgent(
                agent_id="code_agent_001",
                config=code_config
            )
            await app_state.engine.register_agent(
                AgentConfig(
                    agent_id="code_agent_001",
                    capabilities=["code_generation", "code_execution", "code_review"],
                    max_concurrent_tasks=2,
                ),
                instance=code_agent
            )
            app_state.agents["code_agent_001"] = code_agent
            logger.info("Code agent initialized")
        except RuntimeError as e:
            logger.warning(f"Code agent not available - {e}")
    else:
        logger.warning("Code agent not available - missing dependencies")
    
    # Filesystem Agent
    if FileSystemAgent is not None:
        fs_config = {
            "root_dir": str(Path(__file__).parent.parent.parent / "workspace"),
            "allowed_extensions": [".txt", ".py", ".md", ".json", ".csv", ".yaml", ".log"],
        }
        fs_agent = FileSystemAgent(
            agent_id="filesystem_agent_001",
            config=fs_config
        )
        await app_state.engine.register_agent(
            AgentConfig(
                agent_id="filesystem_agent_001",
                capabilities=["file_read", "file_write", "file_list"],
                max_concurrent_tasks=5,
            ),
            instance=fs_agent
        )
        app_state.agents["filesystem_agent_001"] = fs_agent
        logger.info("Filesystem agent initialized")
    else:
        logger.warning("Filesystem agent not available - missing dependencies")
    
    logger.info(f"Initialized {len(app_state.agents)} agents")

async def telemetry_broadcaster():
    """Background task to broadcast telemetry data."""
    import psutil
    
    while True:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            net_io = psutil.net_io_counters()
            
            telemetry = {
                "signalIntegrity": 100.0 - (cpu_percent * 0.1),
                "bandwidth": net_io.bytes_sent / (1024 * 1024),
                "load": cpu_percent,
                "latency": 1.0 + (cpu_percent / 100),
                "timestamp": datetime.now().isoformat(),
            }
            
            await manager.broadcast("telemetry", telemetry)
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"Telemetry broadcast error: {e}")
            await asyncio.sleep(5)

async def status_monitor():
    """Background task to monitor and broadcast system status."""
    while True:
        try:
            if app_state.engine:
                for agent_id, status in app_state.engine.agent_status.items():
                    agent_data = {
                        "id": agent_id,
                        "status": status.value,
                    }
                    await manager.broadcast("agent_update", agent_data)
            
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"Status monitor error: {e}")
            await asyncio.sleep(5)

# ============================================================================
# CREATE APPLICATION
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="ASTRO API",
        description="Autonomous Agent Ecosystem API Server",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Add security middleware (rate limiting, auth, logging, security headers)
    add_security_middleware(app)
    
    # CORS middleware with proper origin validation
    allowed_origins = get_allowed_origins()
    
    # In development, allow all origins; in production, use configured list
    if os.getenv("ASTRO_ENV", "development") == "development":
        allowed_origins = ["*"]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=security_config.cors_allow_methods,
        allow_headers=security_config.cors_allow_headers,
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    )
    
    # Mount static files for web interface
    web_dir = Path(__file__).parent.parent.parent / "web"
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
    
    # Register routes
    register_routes(app)
    
    return app

def register_routes(app: FastAPI):
    """Register all API routes."""
    
    # Initialize and include routers
    from api.routers import system_router
    from api.routers.system import init_router as init_system_router
    init_system_router(app_state, manager)
    app.include_router(system_router)
    
    # Agents router
    from api.routers import agents_router
    from api.routers.agents import init_router as init_agents_router
    init_agents_router(app_state)
    app.include_router(agents_router)
    
    # Workflows router
    from api.routers import workflows_router
    from api.routers.workflows import init_router as init_workflows_router
    init_workflows_router(app_state, manager, WorkflowPriority, Task, Workflow)
    app.include_router(workflows_router)
    
    # Health check routes
    from api.health import router as health_router, init_health_checker
    app.include_router(health_router)
    
    # Initialize health checker with engine and db when available
    if app_state.engine:
        init_health_checker(app_state.engine, app_state.db)
    
    # Metrics endpoint
    from fastapi.responses import PlainTextResponse
    from monitoring.metrics import get_metrics, get_metrics_collector
    
    @app.get("/metrics", response_class=PlainTextResponse, tags=["monitoring"])
    async def metrics():
        """Prometheus metrics endpoint."""
        collector = get_metrics_collector()
        collector.update_uptime()
        return get_metrics().decode('utf-8')
    
    # ========================================================================
    # STATIC FILE ROUTES
    # ========================================================================
    
    @app.get("/")
    async def serve_index():
        """Serve the main web interface."""
        web_dir = Path(__file__).parent.parent.parent / "web"
        index_path = web_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        raise HTTPException(status_code=404, detail="Web interface not found")
    
    @app.get("/styles.css")
    async def serve_styles():
        web_dir = Path(__file__).parent.parent.parent / "web"
        return FileResponse(str(web_dir / "styles.css"), media_type="text/css")
    
    @app.get("/app.js")
    async def serve_app_js():
        web_dir = Path(__file__).parent.parent.parent / "web"
        return FileResponse(str(web_dir / "app.js"), media_type="application/javascript")
    
    @app.get("/chat-ui.css")
    async def serve_chat_css():
        web_dir = Path(__file__).parent.parent.parent / "web"
        return FileResponse(str(web_dir / "chat-ui.css"), media_type="text/css")
    
    @app.get("/chat-app.js")
    async def serve_chat_app_js():
        web_dir = Path(__file__).parent.parent.parent / "web"
        return FileResponse(str(web_dir / "chat-app.js"), media_type="application/javascript")
    
    # New consumer-friendly chat interface
    @app.get("/assistant")
    async def serve_chat_interface():
        """Serve the new consumer-friendly chat interface."""
        web_dir = Path(__file__).parent.parent.parent / "web"
        chat_file = web_dir / "chat.html"
        if chat_file.exists():
            return FileResponse(str(chat_file))
        return FileResponse(str(web_dir / "index.html"))
    
    # SPA fallback routes
    @app.get("/chat")
    @app.get("/workflows")
    @app.get("/vault")
    @app.get("/files")
    async def serve_spa_routes():
        """Serve index.html for SPA routes."""
        web_dir = Path(__file__).parent.parent.parent / "web"
        return FileResponse(str(web_dir / "index.html"))
    
    # ========================================================================
    # HEALTH CHECK ROUTES (for load balancers and monitoring)
    # ========================================================================
    
    @app.get("/health")
    async def health_check():
        """
        Basic health check endpoint.
        
        Returns 200 if the service is running.
        Used by load balancers for basic availability checks.
        """
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    @app.get("/ready")
    async def readiness_check():
        """
        Readiness check endpoint.
        
        Returns 200 if the service is ready to handle requests.
        Checks database connectivity and agent availability.
        """
        checks = {
            "database": False,
            "agents": False,
            "engine": False,
        }
        
        # Check database
        try:
            if app_state.db:
                stats = await app_state.db.get_database_stats_async()
                checks["database"] = True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
        
        # Check agents
        checks["agents"] = len(app_state.agents) > 0
        
        # Check engine
        checks["engine"] = app_state.engine is not None
        
        all_healthy = all(checks.values())
        
        return JSONResponse(
            status_code=200 if all_healthy else 503,
            content={
                "status": "ready" if all_healthy else "not_ready",
                "checks": checks,
                "timestamp": datetime.now().isoformat(),
            }
        )
    
    @app.get("/metrics")
    async def get_metrics():
        """
        Prometheus-compatible metrics endpoint.
        
        Returns system metrics in a structured format.
        """
        import psutil
        
        metrics = {
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
            },
            "application": {
                "uptime_seconds": time.time() - app_state.start_time,
                "agents_count": len(app_state.agents),
                "active_workflows": len(app_state.engine.workflows) if app_state.engine else 0,
                "websocket_connections": len(manager.active_connections),
                "chat_sessions": len(app_state.chat_sessions),
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add agent-specific metrics
        if app_state.engine:
            metrics["agents"] = {}
            for agent_id, status in app_state.engine.agent_status.items():
                metrics["agents"][agent_id] = {
                    "status": status.value,
                    "metrics_count": len(app_state.engine.performance_metrics.get(agent_id, [])),
                }
        
        return metrics
    
    # ========================================================================
    # SYSTEM ROUTES (moved to api/routers/system.py)
    # ========================================================================
    
    # ========================================================================
    # AGENT ROUTES
    # ========================================================================
    
    @app.get("/api/agents", response_model=List[AgentResponse])
    async def get_agents():
        """Get all registered agents."""
        agents = []
        
        agent_meta = {
            "research_agent_001": ("🔬", "Research · 001", "Deepcrawl / Contextual synthesis"),
            "code_agent_001": ("💻", "Code · 001", "Python · QA harness"),
            "filesystem_agent_001": ("📁", "File · 001", "Structured storage / Render"),
        }
        
        for agent_id, agent in app_state.agents.items():
            icon, name, desc = agent_meta.get(agent_id, ("🤖", agent_id, "Agent"))
            status = app_state.engine.agent_status.get(agent_id)
            
            agents.append(AgentResponse(
                id=agent_id,
                name=name,
                icon=icon,
                status=status.value if status else "idle",
                description=desc,
                capabilities=[c.value for c in agent.capabilities] if hasattr(agent, 'capabilities') else [],
            ))
        
        return agents
    
    @app.get("/api/agents/{agent_id}/status")
    async def get_agent_status(agent_id: str):
        """Get status of a specific agent."""
        if agent_id not in app_state.agents:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent = app_state.agents[agent_id]
        status = app_state.engine.agent_status.get(agent_id)
        
        return {
            "id": agent_id,
            "status": status.value if status else "idle",
            "health": await agent.health_check() if hasattr(agent, 'health_check') else {},
        }
    
    # ========================================================================
    # WORKFLOW ROUTES
    # ========================================================================
    
    @app.get("/api/workflows")
    async def get_workflows():
        """Get all workflows."""
        workflows = []
        
        for wf_id, workflow in app_state.engine.workflows.items():
            completed = sum(1 for t in workflow.tasks if t.task_id in app_state.engine.completed_tasks)
            total = len(workflow.tasks)
            
            workflows.append({
                "id": wf_id,
                "name": workflow.name,
                "description": "",
                "status": "completed" if completed == total else "running" if completed > 0 else "pending",
                "priority": workflow.priority.value,
                "progress": int((completed / total) * 100) if total > 0 else 0,
                "tasks": [{"id": t.task_id, "description": t.description} for t in workflow.tasks],
            })
        
        return workflows
    
    @app.post("/api/workflows")
    async def create_workflow(request: WorkflowCreateRequest):
        """Create a new workflow."""
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        priority_map = {
            "low": WorkflowPriority.LOW,
            "medium": WorkflowPriority.MEDIUM,
            "high": WorkflowPriority.HIGH,
            "critical": WorkflowPriority.CRITICAL,
        }
        
        tasks = []
        for i, task_data in enumerate(request.tasks):
            tasks.append(Task(
                task_id=f"{workflow_id}_task_{i}",
                description=task_data.get("description", f"Task {i+1}"),
                required_capabilities=task_data.get("capabilities", []),
                priority=priority_map.get(request.priority, WorkflowPriority.MEDIUM),
            ))
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=request.name,
            tasks=tasks,
            priority=priority_map.get(request.priority, WorkflowPriority.MEDIUM),
        )
        
        await app_state.engine.submit_workflow(workflow)
        
        await manager.broadcast("log", {
            "timestamp": datetime.now().strftime("%H:%M"),
            "type": "workflow",
            "title": "Workflow Created",
            "message": f"Created workflow: {request.name}",
        })
        
        return {"id": workflow_id, "name": request.name, "status": "created"}
    
    @app.get("/api/workflows/{workflow_id}")
    async def get_workflow(workflow_id: str):
        """Get a specific workflow."""
        if workflow_id not in app_state.engine.workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow = app_state.engine.workflows[workflow_id]
        completed = sum(1 for t in workflow.tasks if t.task_id in app_state.engine.completed_tasks)
        
        return {
            "id": workflow_id,
            "name": workflow.name,
            "status": "completed" if completed == len(workflow.tasks) else "running",
            "progress": int((completed / len(workflow.tasks)) * 100) if workflow.tasks else 0,
            "tasks": [
                {
                    "id": t.task_id,
                    "description": t.description,
                    "status": "completed" if t.task_id in app_state.engine.completed_tasks else "pending",
                }
                for t in workflow.tasks
            ],
        }
    
    @app.post("/api/workflows/{workflow_id}/run")
    async def run_workflow(workflow_id: str):
        """Run/resume a workflow."""
        if workflow_id not in app_state.engine.workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow = app_state.engine.workflows[workflow_id]
        
        for task in workflow.tasks:
            if task.task_id not in app_state.engine.completed_tasks:
                await app_state.engine._queue_task(task, workflow.priority)
        
        return {"status": "running", "message": f"Workflow {workflow_id} is now running"}
    
    # ========================================================================
    # COMMAND / CHAT ROUTES
    # ========================================================================
    
    @app.post("/api/command")
    async def execute_command(request: CommandRequest):
        """Execute a natural language command."""
        if not app_state.running:
            raise HTTPException(status_code=400, detail="System is not running. Start the system first.")
        
        # Initialize NL interface if needed
        if not app_state.nl_interface:
            if not app_state.llm_client:
                app_state.llm_client = LLMFactory.create_client(
                    provider=os.getenv("LLM_PROVIDER", "openai"),
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("LLM_BASE_URL"),
                )
            
            app_state.nl_interface = NaturalLanguageInterface(
                engine=app_state.engine,
                llm_client=app_state.llm_client,
                model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            )
        
        try:
            workflow_id = await app_state.nl_interface.process_request(request.command)
            
            await manager.broadcast("log", {
                "timestamp": datetime.now().strftime("%H:%M"),
                "type": "command",
                "title": "Command Executed",
                "message": f"Processing: {request.command[:100]}...",
            })
            
            return {
                "status": "accepted",
                "workflow_id": workflow_id,
                "message": f"Command accepted. Workflow {workflow_id} created.",
            }
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/chat/sessions")
    async def get_chat_sessions():
        """Get all chat sessions."""
        # Try database first
        if app_state.db:
            try:
                sessions = await app_state.db.get_all_sessions_async()
                return sessions
            except Exception as e:
                logger.warning(f"Failed to get sessions from DB: {e}")
        
        # Fallback to in-memory session keys
        return [
            {"id": sid, "title": "Conversation", "createdAt": datetime.now().isoformat()}
            for sid in app_state.chat_sessions.keys()
        ]
    
    @app.get("/api/chat/history/{session_id}")
    async def get_chat_history(session_id: str):
        """Get chat history for a session from database."""
        # Try database first
        if app_state.db:
            try:
                history = await app_state.db.get_chat_history_async(session_id)
                if history:
                    return history
            except Exception as e:
                logger.warning(f"Failed to get chat history from DB: {e}")
        
        # Fallback to in-memory
        return app_state.chat_sessions.get(session_id, [])
    
    @app.post("/api/chat/message")
    async def send_chat_message(request: ChatMessageRequest):
        """Send a chat message and get a response via real LLM inference."""
        session_id = request.session_id
        
        if session_id not in app_state.chat_sessions:
            app_state.chat_sessions[session_id] = []
        
        timestamp = datetime.now().strftime("%H:%M")
        user_message = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": request.message,
            "timestamp": timestamp,
        }
        app_state.chat_sessions[session_id].append(user_message)
        
        # Persist user message to database
        if app_state.db:
            try:
                await app_state.db.save_chat_message_async(
                    session_id, user_message["id"], "user", 
                    request.message, timestamp
                )
            except Exception as e:
                logger.warning(f"Failed to persist user message: {e}")
        
        # Real LLM inference
        response_content = await _generate_chat_response(session_id, request.message)
        
        assistant_timestamp = datetime.now().strftime("%H:%M")
        assistant_message = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": response_content,
            "timestamp": assistant_timestamp,
        }
        app_state.chat_sessions[session_id].append(assistant_message)
        
        # Persist assistant message to database
        if app_state.db:
            try:
                await app_state.db.save_chat_message_async(
                    session_id, assistant_message["id"], "assistant",
                    response_content, assistant_timestamp
                )
            except Exception as e:
                logger.warning(f"Failed to persist assistant message: {e}")
        
        # Broadcast to WebSocket
        await manager.broadcast("chat_message", assistant_message)
        
        return {"message": response_content, "session_id": session_id}
    
    async def _generate_chat_response(session_id: str, user_message: str) -> str:
        """Generate response using Ollama/OpenAI with conversation history."""
        if not app_state.llm_client:
            return "LLM not configured. Please ensure Ollama is running or set an API key."
        
        # Build conversation history (last 20 messages for context)
        history = app_state.chat_sessions.get(session_id, [])[-20:]
        
        system_prompt = """You are ASTRO, an advanced AI assistant with access to specialized agents:
- Research Agent: Web search, content extraction, summarization
- Code Agent: Python code generation and execution (Docker sandboxed)
- FileSystem Agent: File read/write operations in workspace
- Git Agent: Version control operations
- Test Agent: Run test suites
- Analysis Agent: Code linting and static analysis
- Knowledge Agent: Persistent memory storage

You can help with research, coding, file management, and general questions.
Be helpful, accurate, and concise. When tasks require agent capabilities, explain what you're doing."""

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[:-1]:  # Exclude the just-added user message
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})
        
        try:
            model = getattr(app_state, 'llm_model', 'llama3.2')
            response = await app_state.llm_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            return f"I encountered an error: {str(e)}. Please check that Ollama is running."
    
    # ========================================================================
    # KNOWLEDGE ROUTES
    # ========================================================================
    
    @app.get("/api/knowledge")
    async def get_knowledge_items(q: Optional[str] = Query(None)):
        """Get knowledge vault items, optionally filtered by query."""
        # Try database first
        if app_state.db:
            try:
                items = await app_state.db.get_knowledge_items_async(query=q)
                if items:
                    return items
            except Exception as e:
                logger.warning(f"Failed to get knowledge items from DB: {e}")
        
        # Fallback to in-memory
        items = app_state.knowledge_items
        if q:
            q_lower = q.lower()
            items = [
                item for item in items
                if q_lower in item.get("title", "").lower()
                or q_lower in item.get("content", "").lower()
                or any(q_lower in tag.lower() for tag in item.get("tags", []))
            ]
        
        return items
    
    @app.post("/api/knowledge")
    async def create_knowledge_item(request: KnowledgeItemRequest):
        """Create a new knowledge item and persist to database."""
        item_id = str(uuid.uuid4())
        summary = request.content[:200] + "..." if len(request.content) > 200 else request.content
        
        item = {
            "id": item_id,
            "title": request.title,
            "content": request.content,
            "type": request.type,
            "tags": request.tags,
            "created_at": datetime.now().isoformat(),
            "summary": summary,
        }
        
        # Persist to database
        if app_state.db:
            try:
                await app_state.db.save_knowledge_item_async(
                    item_id, request.title, request.content,
                    request.type, request.tags, summary
                )
            except Exception as e:
                logger.warning(f"Failed to persist knowledge item: {e}")
        
        # Also keep in memory for quick access
        app_state.knowledge_items.append(item)
        
        return item
    
    @app.delete("/api/knowledge/{item_id}")
    async def delete_knowledge_item(item_id: str):
        """Delete a knowledge item."""
        # Delete from database
        if app_state.db:
            try:
                deleted = await app_state.db.delete_knowledge_item_async(item_id)
                if deleted:
                    # Also remove from in-memory list
                    app_state.knowledge_items = [
                        i for i in app_state.knowledge_items if i.get("id") != item_id
                    ]
                    return {"status": "deleted", "id": item_id}
            except Exception as e:
                logger.warning(f"Failed to delete knowledge item: {e}")
        
        # Fallback: try in-memory only
        original_len = len(app_state.knowledge_items)
        app_state.knowledge_items = [
            i for i in app_state.knowledge_items if i.get("id") != item_id
        ]
        
        if len(app_state.knowledge_items) < original_len:
            return {"status": "deleted", "id": item_id}
        
        raise HTTPException(status_code=404, detail="Knowledge item not found")
    
    # ========================================================================
    # FILE ROUTES
    # ========================================================================
    
    @app.get("/api/files")
    async def list_files(path: str = Query("/")):
        """List files in the workspace."""
        workspace = Path(__file__).parent.parent.parent / "workspace"
        workspace.mkdir(exist_ok=True)
        
        target_path = workspace / path.lstrip("/")
        
        if not target_path.exists():
            target_path = workspace
        
        # Security: ensure path is within workspace
        try:
            target_path.resolve().relative_to(workspace.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        
        files = []
        
        for item in sorted(target_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            stat = item.stat()
            files.append({
                "name": item.name,
                "path": str(item.relative_to(workspace)),
                "type": "directory" if item.is_dir() else "file",
                "size": stat.st_size if item.is_file() else None,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        
        return files
    
    @app.get("/api/files/content")
    async def get_file_content(path: str = Query(...)):
        """Get contents of a file."""
        workspace = Path(__file__).parent.parent.parent / "workspace"
        file_path = workspace / path.lstrip("/")
        
        # Security: ensure path is within workspace
        try:
            file_path.resolve().relative_to(workspace.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")
        
        try:
            content = file_path.read_text()
            return {"path": path, "content": content}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")
    
    @app.post("/api/files")
    async def create_file(request: FileCreateRequest):
        """Create or update a file."""
        workspace = Path(__file__).parent.parent.parent / "workspace"
        workspace.mkdir(exist_ok=True)
        
        file_path = workspace / request.path.lstrip("/")
        
        # Security: ensure path is within workspace
        try:
            file_path.resolve().relative_to(workspace.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            file_path.write_text(request.content)
            
            await manager.broadcast("file_change", {
                "action": "created" if not file_path.exists() else "updated",
                "path": request.path,
            })
            
            return {"status": "success", "path": request.path}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write file: {e}")
    
    # ========================================================================
    # TELEMETRY ROUTES
    # ========================================================================
    
    @app.get("/api/telemetry", response_model=TelemetryResponse)
    async def get_telemetry():
        """Get current telemetry data."""
        import psutil
        
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        return TelemetryResponse(
            signalIntegrity=100.0 - (cpu_percent * 0.1),
            bandwidth=memory.used / (1024 * 1024 * 1024),
            load=cpu_percent,
            latency=1.0 + (cpu_percent / 100),
            timestamp=datetime.now().isoformat(),
        )
    
    # ========================================================================
    # WEBSOCKET ROUTE
    # ========================================================================
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates with input validation and rate limiting."""
        # Rate limiting per connection
        message_count = 0
        window_start = time.time()
        MAX_MESSAGES_PER_SECOND = 10
        MAX_CONNECTIONS_PER_IP = 5
        MAX_MESSAGE_SIZE = 65536  # 64KB limit
        
        # Check connection limit per IP
        client_ip = websocket.client.host if websocket.client else "unknown"
        ip_connections = sum(1 for ws in manager.active_connections 
                           if manager.connection_metadata.get(ws, {}).get('ip') == client_ip)
        
        if ip_connections >= MAX_CONNECTIONS_PER_IP:
            await websocket.close(code=1008, reason="Too many connections from this IP")
            return
        
        await manager.connect(websocket)
        manager.connection_metadata[websocket] = {'ip': client_ip, 'connected_at': time.time()}
        
        # Send initial state
        await manager.send_to_client(websocket, "system_status", {
            "status": "online" if app_state.running else "ready",
        })
        
        try:
            while True:
                data = await websocket.receive_text()
                
                # Rate limiting check
                current_time = time.time()
                if current_time - window_start >= 1.0:
                    message_count = 0
                    window_start = current_time
                
                message_count += 1
                if message_count > MAX_MESSAGES_PER_SECOND:
                    await manager.send_to_client(websocket, "error", {
                        "message": "Rate limit exceeded",
                        "retry_after": 1.0
                    })
                    continue
                
                # Size validation
                if len(data) > MAX_MESSAGE_SIZE:
                    await manager.send_to_client(websocket, "error", {
                        "message": "Message too large",
                        "max_size": MAX_MESSAGE_SIZE
                    })
                    continue
                
                try:
                    # Parse and validate with Pydantic
                    raw_message = json.loads(data)
                    validated = WSMessage(**raw_message)
                    
                    if validated.type == WSMessageType.PING:
                        await manager.send_to_client(websocket, "pong", {"time": time.time()})
                    elif validated.type == WSMessageType.SUBSCRIBE:
                        pass  # Handle subscription
                    else:
                        logger.debug(f"Received message type: {validated.type}")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {data[:100]}")
                    await manager.send_to_client(websocket, "error", {"message": "Invalid JSON"})
                except Exception as e:
                    logger.warning(f"WebSocket validation error: {e}")
                    await manager.send_to_client(websocket, "error", {"message": "Invalid message format"})
                    
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            manager.disconnect(websocket)

# ============================================================================
# SERVER RUNNER
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server."""
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    uvicorn.run(
        "api.server:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )

# Create app instance for uvicorn
app = create_app()

if __name__ == "__main__":
    run_server()
