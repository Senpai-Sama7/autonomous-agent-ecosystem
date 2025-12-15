"""
Database Persistence Layer

REFACTORED: Native async database operations using aiosqlite.
- True async I/O without blocking the event loop
- Connection pooling for improved performance
- Sync API marked DEPRECATED - use async methods in production

SECURITY: Sync methods are deprecated for production use due to potential
blocking/race conditions. They remain for CLI scripts and testing only.
"""

import json
import sqlite3
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import logging

import warnings

# Import aiosqlite for native async SQLite
try:
    import aiosqlite

    HAS_AIOSQLITE = True
except ImportError:
    aiosqlite = None
    HAS_AIOSQLITE = False
    logging.warning(
        "aiosqlite not installed. Async DB operations will use thread fallback."
    )


def _deprecation_warning(method_name: str):
    """Emit deprecation warning for sync methods in production context"""
    warnings.warn(
        f"{method_name}() is deprecated for production use. "
        f"Use {method_name}_async() instead to avoid blocking the event loop.",
        DeprecationWarning,
        stacklevel=3,
    )


logger = logging.getLogger("DatabaseManager")


class ConnectionPool:
    """Simple async connection pool for SQLite."""

    def __init__(self, db_path: str, max_connections: int = 5, timeout: float = 30.0):
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=max_connections)
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the connection pool."""
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            for _ in range(self.max_connections):
                conn = await aiosqlite.connect(self.db_path, timeout=self.timeout)
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute(f"PRAGMA busy_timeout={int(self.timeout * 1000)}")
                await self._pool.put(conn)
            self._initialized = True
            logger.debug(
                f"Connection pool initialized with {self.max_connections} connections"
            )

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self._initialized:
            await self.initialize()
        conn = await asyncio.wait_for(self._pool.get(), timeout=self.timeout)
        try:
            yield conn
        finally:
            await self._pool.put(conn)

    async def close(self):
        """Close all connections in the pool."""
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()
        self._initialized = False


class DatabaseManager:
    def __init__(
        self,
        db_path: str = "ecosystem.db",
        connection_timeout: float = 30.0,
        pool_size: int = 5,
    ):
        self.db_path = db_path
        self.connection_timeout = connection_timeout
        self._async_initialized = False
        self._sync_initialized = False
        self._closed = False
        self._pool: Optional[ConnectionPool] = None
        self._pool_size = pool_size
        # Note: Database initialization moved to async_init() to avoid blocking

    async def async_init(self):
        """Async initialization to avoid blocking event loop"""
        if not self._async_initialized:
            await self._init_db_async()
            if HAS_AIOSQLITE:
                self._pool = ConnectionPool(
                    self.db_path, self._pool_size, self.connection_timeout
                )
                await self._pool.initialize()
            self._async_initialized = True

    async def _init_db_async(self):
        """Initialize database schema asynchronously"""
        if not HAS_AIOSQLITE:
            # Fallback to sync for initialization only
            self._init_db_sync()
            return

        async with aiosqlite.connect(self.db_path) as db:
            await self._create_tables(db)

    def _init_db_sync(self):
        """Synchronous fallback for database initialization"""
        _deprecation_warning("_init_db_sync")
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        self._create_tables_sync(c)
        conn.commit()
        conn.close()
        self._sync_initialized = True

    def _ensure_sync_init(self) -> None:
        """Ensure synchronous schema is initialized before direct writes."""
        with self._sync_init_lock:
            if self._sync_initialized:
                return
            if self._closed:
                raise RuntimeError("DatabaseManager has been closed")
            self._init_db_sync()

    async def _create_tables(self, db):
        """Create tables asynchronously"""
        # Agents table
        await db.execute(
            """CREATE TABLE IF NOT EXISTS agents
                         (agent_id TEXT PRIMARY KEY,
                          config TEXT,
                          state TEXT,
                          reliability_score REAL,
                          last_updated TIMESTAMP)"""
        )

        # Workflows table
        await db.execute(
            """CREATE TABLE IF NOT EXISTS workflows
                         (workflow_id TEXT PRIMARY KEY,
                          name TEXT,
                          status TEXT,
                          priority TEXT,
                          created_at TIMESTAMP,
                          completed_at TIMESTAMP)"""
        )

        # Tasks table
        await db.execute(
            """CREATE TABLE IF NOT EXISTS tasks
                         (task_id TEXT PRIMARY KEY,
                          workflow_id TEXT,
                          description TEXT,
                          status TEXT,
                          assigned_agent TEXT,
                          result TEXT,
                          created_at TIMESTAMP,
                          completed_at TIMESTAMP)"""
        )

        # Metrics table
        await db.execute(
            """CREATE TABLE IF NOT EXISTS metrics
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          agent_id TEXT,
                          metric_name TEXT,
                          metric_value REAL,
                          timestamp TIMESTAMP)"""
        )

        # Chat sessions table
        await db.execute(
            """CREATE TABLE IF NOT EXISTS chat_sessions
                         (session_id TEXT PRIMARY KEY,
                          created_at TIMESTAMP,
                          last_active TIMESTAMP,
                          metadata TEXT)"""
        )

        # Chat messages table
        await db.execute(
            """CREATE TABLE IF NOT EXISTS chat_messages
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          message_id TEXT,
                          session_id TEXT,
                          role TEXT,
                          content TEXT,
                          timestamp TIMESTAMP,
                          FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id))"""
        )

        # Knowledge items table
        await db.execute(
            """CREATE TABLE IF NOT EXISTS knowledge_items
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          title TEXT,
                          content TEXT,
                          type TEXT,
                          tags TEXT,
                          summary TEXT,
                          embedding BLOB,
                          metadata TEXT,
                          created_at TIMESTAMP,
                          updated_at TIMESTAMP)"""
        )

        # Learning patterns table
        await db.execute(
            """CREATE TABLE IF NOT EXISTS learning_patterns
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          pattern_type TEXT,
                          pattern_data TEXT,
                          frequency INTEGER,
                          success_rate REAL,
                          last_seen TIMESTAMP)"""
        )

        # Learning metadata table
        await db.execute(
            """CREATE TABLE IF NOT EXISTS learning_metadata
                         (key TEXT PRIMARY KEY,
                          value TEXT,
                          updated_at TIMESTAMP)"""
        )

        await db.commit()

    def _create_tables_sync(self, c):
        """Create tables synchronously"""
        # Agents table
        c.execute(
            """CREATE TABLE IF NOT EXISTS agents
                     (agent_id TEXT PRIMARY KEY,
                      config TEXT,
                      state TEXT,
                      reliability_score REAL,
                      last_updated TIMESTAMP)"""
        )

        # Workflows table
        c.execute(
            """CREATE TABLE IF NOT EXISTS workflows
                     (workflow_id TEXT PRIMARY KEY,
                      name TEXT,
                      status TEXT,
                      priority TEXT,
                      created_at TIMESTAMP,
                      completed_at TIMESTAMP)"""
        )

        # Tasks table
        c.execute(
            """CREATE TABLE IF NOT EXISTS tasks
                     (task_id TEXT PRIMARY KEY,
                      workflow_id TEXT,
                      description TEXT,
                      status TEXT,
                      assigned_agent TEXT,
                      result TEXT,
                      created_at TIMESTAMP,
                      completed_at TIMESTAMP)"""
        )

        # Metrics table
        c.execute(
            """CREATE TABLE IF NOT EXISTS metrics
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      agent_id TEXT,
                      metric_name TEXT,
                      metric_value REAL,
                      timestamp TIMESTAMP)"""
        )

        # Chat sessions table
        c.execute(
            """CREATE TABLE IF NOT EXISTS chat_sessions
                     (session_id TEXT PRIMARY KEY,
                      created_at TIMESTAMP,
                      last_activity TIMESTAMP,
                      metadata TEXT)"""
        )

        # Chat messages table
        c.execute(
            """CREATE TABLE IF NOT EXISTS chat_messages
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      message_id TEXT,
                      session_id TEXT,
                      role TEXT,
                      content TEXT,
                      timestamp TIMESTAMP,
                      FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id))"""
        )

        # Knowledge items table
        c.execute(
            """CREATE TABLE IF NOT EXISTS knowledge_items
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      title TEXT,
                      content TEXT,
                      type TEXT,
                      tags TEXT,
                      summary TEXT,
                      embedding BLOB,
                      metadata TEXT,
                      created_at TIMESTAMP,
                      updated_at TIMESTAMP)"""
        )

        # Learning patterns table
        c.execute(
            """CREATE TABLE IF NOT EXISTS learning_patterns
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      pattern_type TEXT,
                      pattern_data TEXT,
                      frequency INTEGER,
                      success_rate REAL,
                      last_seen TIMESTAMP)"""
        )

        # Learning metadata table
        c.execute(
            """CREATE TABLE IF NOT EXISTS learning_metadata
                     (key TEXT PRIMARY KEY,
                      value TEXT,
                      updated_at TIMESTAMP)"""
        )

    def _init_db(self):
        """Initialize database schema (deprecated - use _init_db_async)"""
        _deprecation_warning("_init_db")
        self._init_db_sync()

        # Agents table
        c.execute(
            """CREATE TABLE IF NOT EXISTS agents
                     (agent_id TEXT PRIMARY KEY,
                      config TEXT,
                      state TEXT,
                      reliability_score REAL,
                      last_updated TIMESTAMP)"""
        )

        # Workflows table
        c.execute(
            """CREATE TABLE IF NOT EXISTS workflows
                     (workflow_id TEXT PRIMARY KEY,
                      name TEXT,
                      status TEXT,
                      priority TEXT,
                      created_at TIMESTAMP)"""
        )

        # Tasks table
        c.execute(
            """CREATE TABLE IF NOT EXISTS tasks
                     (task_id TEXT PRIMARY KEY,
                      workflow_id TEXT,
                      description TEXT,
                      status TEXT,
                      assigned_agent TEXT,
                      result TEXT,
                      created_at TIMESTAMP,
                      completed_at TIMESTAMP,
                      FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id))"""
        )

        # Metrics table
        c.execute(
            """CREATE TABLE IF NOT EXISTS metrics
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      agent_id TEXT,
                      metric_type TEXT,
                      value REAL,
                      timestamp TIMESTAMP)"""
        )

        # Learning patterns table (for recursive_learning.py)
        c.execute(
            """CREATE TABLE IF NOT EXISTS learning_patterns
                     (pattern_id TEXT PRIMARY KEY,
                      pattern_type TEXT,
                      conditions TEXT,
                      action TEXT,
                      expected_outcome TEXT,
                      confidence REAL,
                      usage_count INTEGER DEFAULT 0,
                      success_count INTEGER DEFAULT 0,
                      last_used TIMESTAMP,
                      created_at TIMESTAMP)"""
        )

        # Learning metadata table
        c.execute(
            """CREATE TABLE IF NOT EXISTS learning_metadata
                     (key TEXT PRIMARY KEY,
                      value TEXT,
                      updated_at TIMESTAMP)"""
        )

        # Chat sessions table
        c.execute(
            """CREATE TABLE IF NOT EXISTS chat_sessions
                     (session_id TEXT PRIMARY KEY,
                      created_at TIMESTAMP,
                      last_active TIMESTAMP)"""
        )

        # Chat messages table
        c.execute(
            """CREATE TABLE IF NOT EXISTS chat_messages
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id TEXT,
                      message_id TEXT UNIQUE,
                      role TEXT,
                      content TEXT,
                      timestamp TIMESTAMP,
                      FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id))"""
        )

        # Knowledge items table
        c.execute(
            """CREATE TABLE IF NOT EXISTS knowledge_items
                     (id TEXT PRIMARY KEY,
                      title TEXT,
                      content TEXT,
                      type TEXT,
                      tags TEXT,
                      summary TEXT,
                      created_at TIMESTAMP,
                      updated_at TIMESTAMP)"""
        )

        # Create Indices
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_workflow_id ON tasks(workflow_id)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_assigned_agent ON tasks(assigned_agent)"
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_agent_id ON metrics(agent_id)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_items_type ON knowledge_items(type)"
        )

        conn.commit()
        conn.close()

    # ========== ASYNC INITIALIZATION ==========

    async def _ensure_async_init(self):
        """
        Ensure async schema is initialized (idempotent).

        MUST be called before any async operation. This method:
        - Initializes connection pool
        - Enables WAL mode for better concurrency
        - Is idempotent (safe to call multiple times)
        """
        if self._async_initialized:
            return

        if self._closed:
            raise RuntimeError("DatabaseManager has been closed")

        if HAS_AIOSQLITE:
            try:
                self._pool = ConnectionPool(
                    self.db_path, self._pool_size, self.connection_timeout
                )
                await self._pool.initialize()
                self._async_initialized = True
                logger.debug("Async database initialized with connection pool")
            except Exception as e:
                logger.error(f"Failed to initialize async database: {e}")
                raise
        else:
            logger.warning(
                "aiosqlite not available - async operations will use thread pool"
            )
            self._async_initialized = True

    async def close_async(self):
        """Close database manager and connection pool."""
        if self._pool:
            await self._pool.close()
        self._closed = True

    async def close(self):
        """Alias for close_async for consistency."""
        await self.close_async()

    # ========== SYNC HELPERS ==========

    def _execute_write(self, sql: str, params: tuple):
        """Execute a sync write operation"""
        self._ensure_sync_init()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(sql, params)
                conn.commit()
        except Exception as e:
            logger.error(f"DB Write Error: {e}")
            raise

    # ========== SYNC API (DEPRECATED for production) ==========
    # These methods block the event loop and may cause race conditions.
    # Use async versions in production code.

    def save_agent(self, agent_id: str, config: Dict, state: str, reliability: float):
        """Sync save agent (blocks caller) - DEPRECATED for production"""
        _deprecation_warning("save_agent")
        sql = """INSERT OR REPLACE INTO agents
                 (agent_id, config, state, reliability_score, last_updated)
                 VALUES (?, ?, ?, ?, ?)"""
        params = (
            agent_id,
            json.dumps(config),
            state,
            reliability,
            datetime.now().isoformat(),
        )
        self._execute_write(sql, params)

    def save_workflow(self, workflow_id: str, name: str, status: str, priority: str):
        """Sync save workflow (blocks caller) - DEPRECATED for production"""
        _deprecation_warning("save_workflow")
        sql = """INSERT OR REPLACE INTO workflows
                 (workflow_id, name, status, priority, created_at)
                 VALUES (?, ?, ?, ?, ?)"""
        params = (workflow_id, name, status, priority, datetime.now().isoformat())
        self._execute_write(sql, params)

    def save_task(
        self,
        task_id: str,
        workflow_id: str,
        description: str,
        status: str,
        assigned_agent: Optional[str] = None,
        result: Optional[Dict] = None,
    ):
        """Sync save task (blocks caller) - DEPRECATED for production"""
        _deprecation_warning("save_task")
        sql = """INSERT OR REPLACE INTO tasks
                 (task_id, workflow_id, description, status, assigned_agent, result, created_at, completed_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
        now = datetime.now().isoformat()
        params = (
            task_id,
            workflow_id,
            description,
            status,
            assigned_agent,
            json.dumps(result) if result else None,
            now,
            now if status == "completed" else None,
        )
        self._execute_write(sql, params)

    def log_metric(self, agent_id: str, metric_type: str, value: float):
        """Sync log metric (blocks caller) - DEPRECATED for production"""
        _deprecation_warning("log_metric")
        sql = """INSERT INTO metrics (agent_id, metric_type, value, timestamp)
                 VALUES (?, ?, ?, ?)"""
        params = (agent_id, metric_type, value, datetime.now().isoformat())
        self._execute_write(sql, params)

    # ========== ASYNC API (Native aiosqlite) ==========

    async def save_agent_async(
        self, agent_id: str, config: Dict, state: str, reliability: float
    ):
        """Async save agent using connection pool"""
        await self._ensure_async_init()
        sql = """INSERT OR REPLACE INTO agents
                 (agent_id, config, state, reliability_score, last_updated)
                 VALUES (?, ?, ?, ?, ?)"""
        params = (
            agent_id,
            json.dumps(config),
            state,
            reliability,
            datetime.now().isoformat(),
        )

        if HAS_AIOSQLITE and self._pool:
            async with self._pool.acquire() as db:
                await db.execute(sql, params)
                await db.commit()
        else:
            await asyncio.to_thread(
                self.save_agent, agent_id, config, state, reliability
            )

    async def save_workflow_async(
        self, workflow_id: str, name: str, status: str, priority: str
    ):
        """Async save workflow using connection pool"""
        await self._ensure_async_init()
        sql = """INSERT OR REPLACE INTO workflows
                 (workflow_id, name, status, priority, created_at)
                 VALUES (?, ?, ?, ?, ?)"""
        params = (workflow_id, name, status, priority, datetime.now().isoformat())

        if HAS_AIOSQLITE and self._pool:
            async with self._pool.acquire() as db:
                await db.execute(sql, params)
                await db.commit()
        else:
            await asyncio.to_thread(
                self.save_workflow, workflow_id, name, status, priority
            )

    async def save_task_async(
        self,
        task_id: str,
        workflow_id: str,
        description: str,
        status: str,
        assigned_agent: Optional[str] = None,
        result: Optional[Dict] = None,
    ):
        """Async save task using connection pool"""
        await self._ensure_async_init()
        sql = """INSERT OR REPLACE INTO tasks
                 (task_id, workflow_id, description, status, assigned_agent, result, created_at, completed_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
        params = (
            task_id,
            workflow_id,
            description,
            status,
            assigned_agent,
            json.dumps(result) if result else None,
            datetime.now().isoformat(),
            datetime.now().isoformat() if status == "completed" else None,
        )

        if HAS_AIOSQLITE and self._pool:
            async with self._pool.acquire() as db:
                await db.execute(sql, params)
                await db.commit()
        else:
            await asyncio.to_thread(
                self.save_task,
                task_id,
                workflow_id,
                description,
                status,
                assigned_agent,
                result,
            )

    async def log_metric_async(self, agent_id: str, metric_type: str, value: float):
        """Async log metric using native aiosqlite"""
        await self._ensure_async_init()
        sql = """INSERT INTO metrics (agent_id, metric_type, value, timestamp)
                 VALUES (?, ?, ?, ?)"""
        params = (agent_id, metric_type, value, datetime.now().isoformat())

        if HAS_AIOSQLITE:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(sql, params)
                await db.commit()
        else:
            await asyncio.to_thread(self.log_metric, agent_id, metric_type, value)

    async def get_recent_metrics_async(
        self, agent_id: str, metric_type: str, limit: int = 100
    ) -> List[Dict]:
        """Async fetch recent metrics using native aiosqlite"""
        await self._ensure_async_init()
        sql = """SELECT value, timestamp FROM metrics
                 WHERE agent_id = ? AND metric_type = ?
                 ORDER BY timestamp DESC LIMIT ?"""
        params = (agent_id, metric_type, limit)

        if HAS_AIOSQLITE:
            async with aiosqlite.connect(
                self.db_path, timeout=self.connection_timeout
            ) as db:
                async with db.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()
                    return [{"value": row[0], "timestamp": row[1]} for row in rows]
        else:
            # Fallback - run in thread to avoid blocking
            def _sync_read():
                with sqlite3.connect(
                    self.db_path, timeout=self.connection_timeout
                ) as conn:
                    cursor = conn.execute(sql, params)
                    return cursor.fetchall()

            rows = await asyncio.to_thread(_sync_read)
            return [{"value": row[0], "timestamp": row[1]} for row in rows]

    async def get_agent_stats_async(self, agent_id: str) -> Dict[str, Any]:
        """Async fetch aggregated stats using native aiosqlite"""
        await self._ensure_async_init()
        sql = """SELECT metric_type, AVG(value) as avg_val, COUNT(*) as count
                 FROM metrics WHERE agent_id = ?
                 GROUP BY metric_type"""
        params = (agent_id,)

        if HAS_AIOSQLITE:
            async with aiosqlite.connect(
                self.db_path, timeout=self.connection_timeout
            ) as db:
                async with db.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()
                    return {
                        row[0]: {"avg": row[1], "count": row[2]}
                        for row in rows
                        if row[0]
                    }
        else:
            # Fallback - run in thread to avoid blocking
            def _sync_read():
                with sqlite3.connect(
                    self.db_path, timeout=self.connection_timeout
                ) as conn:
                    cursor = conn.execute(sql, params)
                    return cursor.fetchall()

            rows = await asyncio.to_thread(_sync_read)
            return {row[0]: {"avg": row[1], "count": row[2]} for row in rows if row[0]}

    async def get_all_agents_async(self) -> List[Dict]:
        """Async fetch all registered agents"""
        await self._ensure_async_init()
        sql = "SELECT agent_id, config, state, reliability_score, last_updated FROM agents"

        def _parse_row(row) -> Dict:
            """Safely parse a row with error handling for JSON"""
            try:
                config = json.loads(row[1]) if row[1] else {}
            except (json.JSONDecodeError, TypeError):
                config = {}
                logger.warning(f"Failed to parse config for agent {row[0]}")
            return {
                "agent_id": row[0],
                "config": config,
                "state": row[2],
                "reliability_score": row[3],
                "last_updated": row[4],
            }

        if HAS_AIOSQLITE:
            async with aiosqlite.connect(
                self.db_path, timeout=self.connection_timeout
            ) as db:
                async with db.execute(sql) as cursor:
                    rows = await cursor.fetchall()
                    return [_parse_row(row) for row in rows]
        else:
            # Fallback - run in thread to avoid blocking
            def _sync_read():
                with sqlite3.connect(
                    self.db_path, timeout=self.connection_timeout
                ) as conn:
                    cursor = conn.execute(sql)
                    return cursor.fetchall()

            rows = await asyncio.to_thread(_sync_read)
            return [_parse_row(row) for row in rows]

    def close(self):
        """Cleanup resources and mark as closed"""
        self._closed = True
        logger.info("Database manager closed")

    async def prune_old_metrics_async(self, days: int = 30) -> int:
        """Remove metrics older than specified days. Returns count of deleted rows."""
        await self._ensure_async_init()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        sql = "DELETE FROM metrics WHERE timestamp < ?"
        params = (cutoff,)

        if HAS_AIOSQLITE:
            async with aiosqlite.connect(
                self.db_path, timeout=self.connection_timeout
            ) as db:
                cursor = await db.execute(sql, params)
                await db.commit()
                return cursor.rowcount
        else:

            def _sync_delete():
                with sqlite3.connect(
                    self.db_path, timeout=self.connection_timeout
                ) as conn:
                    cursor = conn.execute(sql, params)
                    conn.commit()
                    return cursor.rowcount

            return await asyncio.to_thread(_sync_delete)

    async def get_database_stats_async(self) -> Dict[str, Any]:
        """Get database statistics for monitoring"""
        await self._ensure_async_init()
        stats = {}

        tables = ["agents", "workflows", "tasks", "metrics"]
        for table in tables:
            sql = f"SELECT COUNT(*) FROM {table}"
            if HAS_AIOSQLITE:
                async with aiosqlite.connect(
                    self.db_path, timeout=self.connection_timeout
                ) as db:
                    async with db.execute(sql) as cursor:
                        row = await cursor.fetchone()
                        stats[f"{table}_count"] = row[0] if row else 0
            else:

                def _sync_count(t=table):
                    with sqlite3.connect(
                        self.db_path, timeout=self.connection_timeout
                    ) as conn:
                        cursor = conn.execute(f"SELECT COUNT(*) FROM {t}")
                        row = cursor.fetchone()
                        return row[0] if row else 0

                stats[f"{table}_count"] = await asyncio.to_thread(_sync_count)

        return stats

    # ========== LEARNING PATTERN PERSISTENCE ==========

    def save_learning_pattern(
        self,
        pattern_id: str,
        pattern_type: str,
        conditions: List[Dict],
        action: str,
        expected_outcome: str,
        confidence: float,
        usage_count: int = 0,
        success_count: int = 0,
        last_used: str = None,
    ):
        """Save a learning pattern to the database"""
        sql = """INSERT OR REPLACE INTO learning_patterns
                 (pattern_id, pattern_type, conditions, action, expected_outcome,
                  confidence, usage_count, success_count, last_used, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        now = datetime.now().isoformat()
        params = (
            pattern_id,
            pattern_type,
            json.dumps(conditions),
            action,
            expected_outcome,
            confidence,
            usage_count,
            success_count,
            last_used or now,
            now,
        )
        self._execute_write(sql, params)

    def load_learning_patterns(self) -> List[Dict[str, Any]]:
        """Load all learning patterns from the database"""
        try:
            with sqlite3.connect(self.db_path, timeout=self.connection_timeout) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM learning_patterns")
                patterns = []
                for row in cursor.fetchall():
                    patterns.append(
                        {
                            "pattern_id": row["pattern_id"],
                            "pattern_type": row["pattern_type"],
                            "conditions": (
                                json.loads(row["conditions"])
                                if row["conditions"]
                                else []
                            ),
                            "action": row["action"],
                            "expected_outcome": row["expected_outcome"],
                            "confidence": row["confidence"],
                            "usage_count": row["usage_count"],
                            "success_count": row["success_count"],
                            "last_used": row["last_used"],
                        }
                    )
                return patterns
        except Exception as e:
            logger.error(f"Failed to load learning patterns: {e}")
            return []

    def save_learning_metadata(self, key: str, value: Any):
        """Save learning metadata (e.g., iteration count)"""
        sql = """INSERT OR REPLACE INTO learning_metadata (key, value, updated_at)
                 VALUES (?, ?, ?)"""
        params = (key, json.dumps(value), datetime.now().isoformat())
        self._execute_write(sql, params)

    def load_learning_metadata(self, key: str, default: Any = None) -> Any:
        """Load learning metadata by key"""
        try:
            with sqlite3.connect(self.db_path, timeout=self.connection_timeout) as conn:
                cursor = conn.execute(
                    "SELECT value FROM learning_metadata WHERE key = ?", (key,)
                )
                row = cursor.fetchone()
                return json.loads(row[0]) if row else default
        except Exception as e:
            logger.error(f"Failed to load learning metadata '{key}': {e}")
            return default

    def backup(self, backup_path: str):
        """Create a backup of the database"""
        import shutil

        try:
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

    def restore(self, backup_path: str):
        """Restore database from backup"""
        import shutil

        try:
            shutil.copy2(backup_path, self.db_path)
            logger.info(f"Database restored from {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    # ========== CHAT SESSION PERSISTENCE ==========

    async def save_chat_message_async(
        self, session_id: str, message_id: str, role: str, content: str, timestamp: str
    ) -> bool:
        """Save a chat message to the database."""
        await self._ensure_async_init()

        # Ensure session exists
        session_sql = """INSERT OR IGNORE INTO chat_sessions (session_id, created_at, last_active)
                        VALUES (?, ?, ?)"""
        message_sql = """INSERT OR REPLACE INTO chat_messages
                        (session_id, message_id, role, content, timestamp)
                        VALUES (?, ?, ?, ?, ?)"""
        now = datetime.now().isoformat()

        if HAS_AIOSQLITE:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(session_sql, (session_id, now, now))
                await db.execute(
                    message_sql, (session_id, message_id, role, content, timestamp)
                )
                await db.execute(
                    "UPDATE chat_sessions SET last_active = ? WHERE session_id = ?",
                    (now, session_id),
                )
                await db.commit()
                return True
        else:

            def _sync_save():
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(session_sql, (session_id, now, now))
                    conn.execute(
                        message_sql, (session_id, message_id, role, content, timestamp)
                    )
                    conn.execute(
                        "UPDATE chat_sessions SET last_active = ? WHERE session_id = ?",
                        (now, session_id),
                    )
                    conn.commit()
                    return True

            return await asyncio.to_thread(_sync_save)

    async def get_chat_history_async(
        self, session_id: str, limit: int = 100
    ) -> List[Dict]:
        """Get chat history for a session."""
        await self._ensure_async_init()
        sql = """SELECT message_id, role, content, timestamp
                 FROM chat_messages WHERE session_id = ?
                 ORDER BY id ASC LIMIT ?"""
        params = (session_id, limit)

        if HAS_AIOSQLITE:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()
                    return [
                        {
                            "id": row[0],
                            "role": row[1],
                            "content": row[2],
                            "timestamp": row[3],
                        }
                        for row in rows
                    ]
        else:

            def _sync_read():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(sql, params)
                    return cursor.fetchall()

            rows = await asyncio.to_thread(_sync_read)
            return [
                {"id": row[0], "role": row[1], "content": row[2], "timestamp": row[3]}
                for row in rows
            ]

    async def get_all_sessions_async(self) -> List[Dict]:
        """Get all chat sessions."""
        await self._ensure_async_init()
        sql = "SELECT session_id, created_at, last_active FROM chat_sessions ORDER BY last_active DESC"

        if HAS_AIOSQLITE:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(sql) as cursor:
                    rows = await cursor.fetchall()
                    return [
                        {
                            "session_id": row[0],
                            "created_at": row[1],
                            "last_active": row[2],
                        }
                        for row in rows
                    ]
        else:

            def _sync_read():
                with sqlite3.connect(self.db_path) as conn:
                    return conn.execute(sql).fetchall()

            rows = await asyncio.to_thread(_sync_read)
            return [
                {"session_id": row[0], "created_at": row[1], "last_active": row[2]}
                for row in rows
            ]

    # ========== KNOWLEDGE ITEM PERSISTENCE ==========

    async def save_knowledge_item_async(
        self,
        item_id: str,
        title: str,
        content: str,
        item_type: str,
        tags: List[str],
        summary: str,
    ) -> bool:
        """Save a knowledge item to the database."""
        await self._ensure_async_init()
        sql = """INSERT OR REPLACE INTO knowledge_items
                (id, title, content, type, tags, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
        now = datetime.now().isoformat()
        tags_json = json.dumps(tags)
        params = (
            item_id,
            title,
            content,
            item_type,
            tags_json,
            summary,
            now,
            now,
        )

        if HAS_AIOSQLITE:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(sql, params)
                await db.commit()
                return True
        else:

            def _sync_save():
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(sql, params)
                    conn.commit()
                    return True

            return await asyncio.to_thread(_sync_save)

    async def get_knowledge_items_async(
        self,
        query: Optional[str] = None,
        item_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get knowledge items with optional filtering."""
        await self._ensure_async_init()

        sql = "SELECT id, title, content, type, tags, summary, created_at FROM knowledge_items"
        conditions = []
        params = []

        if query:
            conditions.append("(title LIKE ? OR content LIKE ? OR tags LIKE ?)")
            query_pattern = f"%{query}%"
            params.extend([query_pattern, query_pattern, query_pattern])

        if item_type:
            conditions.append("type = ?")
            params.append(item_type)

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        def _parse_row(row) -> Dict:
            try:
                tags = json.loads(row[4]) if row[4] else []
            except (json.JSONDecodeError, TypeError):
                tags = []
            return {
                "id": row[0],
                "title": row[1],
                "content": row[2],
                "type": row[3],
                "tags": tags,
                "summary": row[5],
                "created_at": row[6],
            }

        if HAS_AIOSQLITE:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()
                    return [_parse_row(row) for row in rows]
        else:

            def _sync_read():
                with sqlite3.connect(self.db_path) as conn:
                    return conn.execute(sql, params).fetchall()

            rows = await asyncio.to_thread(_sync_read)
            return [_parse_row(row) for row in rows]

    async def delete_knowledge_item_async(self, item_id: str) -> bool:
        """Delete a knowledge item."""
        await self._ensure_async_init()
        sql = "DELETE FROM knowledge_items WHERE id = ?"

        if HAS_AIOSQLITE:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(sql, (item_id,))
                await db.commit()
                return cursor.rowcount > 0
        else:

            def _sync_delete():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(sql, (item_id,))
                    conn.commit()
                    return cursor.rowcount > 0

            return await asyncio.to_thread(_sync_delete)


# CLI interface
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Database Management CLI")
    parser.add_argument(
        "command", choices=["backup", "restore"], help="Command to execute"
    )
    parser.add_argument("--db", default="ecosystem.db", help="Database file path")
    parser.add_argument("--file", required=True, help="Backup file path")

    args = parser.parse_args()

    db = DatabaseManager(args.db)

    if args.command == "backup":
        if db.backup(args.file):
            print(f"✅ Backup created: {args.file}")
            sys.exit(0)
        else:
            print(f"❌ Backup failed")
            sys.exit(1)
    elif args.command == "restore":
        if db.restore(args.file):
            print(f"✅ Database restored from: {args.file}")
            sys.exit(0)
        else:
            print(f"❌ Restore failed")
            sys.exit(1)
