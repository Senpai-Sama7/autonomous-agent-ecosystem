"""
Database Persistence Layer
"""
import json
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from dataclasses import asdict

logger = logging.getLogger("DatabaseManager")

class DatabaseManager:
    def __init__(self, db_path: str = "ecosystem.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Agents table
        c.execute('''CREATE TABLE IF NOT EXISTS agents
                     (agent_id TEXT PRIMARY KEY, 
                      config TEXT,
                      state TEXT,
                      reliability_score REAL,
                      last_updated TIMESTAMP)''')
                      
        # Workflows table
        c.execute('''CREATE TABLE IF NOT EXISTS workflows
                     (workflow_id TEXT PRIMARY KEY,
                      name TEXT,
                      status TEXT,
                      priority TEXT,
                      created_at TIMESTAMP)''')
                      
        # Tasks table
        c.execute('''CREATE TABLE IF NOT EXISTS tasks
                     (task_id TEXT PRIMARY KEY,
                      workflow_id TEXT,
                      description TEXT,
                      status TEXT,
                      assigned_agent TEXT,
                      result TEXT,
                      created_at TIMESTAMP,
                      completed_at TIMESTAMP,
                      FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id))''')
                      
        # Metrics table
        c.execute('''CREATE TABLE IF NOT EXISTS metrics
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      agent_id TEXT,
                      metric_type TEXT,
                      value REAL,
                      timestamp TIMESTAMP)''')
                      
        # Create Indices
        c.execute('CREATE INDEX IF NOT EXISTS idx_tasks_workflow_id ON tasks(workflow_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_tasks_assigned_agent ON tasks(assigned_agent)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_metrics_agent_id ON metrics(agent_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)')
                      
        conn.commit()
        conn.close()

    def save_agent(self, agent_id: str, config: Dict, state: str, reliability: float):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO agents 
                     (agent_id, config, state, reliability_score, last_updated)
                     VALUES (?, ?, ?, ?, ?)''',
                  (agent_id, json.dumps(config), state, reliability, datetime.now()))
        conn.commit()
        conn.close()

    def save_workflow(self, workflow_id: str, name: str, status: str, priority: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO workflows 
                     (workflow_id, name, status, priority, created_at)
                     VALUES (?, ?, ?, ?, ?)''',
                  (workflow_id, name, status, priority, datetime.now()))
        conn.commit()
        conn.close()

    def save_task(self, task_id: str, workflow_id: str, description: str, status: str, 
                  assigned_agent: Optional[str] = None, result: Optional[Dict] = None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO tasks 
                     (task_id, workflow_id, description, status, assigned_agent, result, created_at, completed_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (task_id, workflow_id, description, status, assigned_agent, 
                   json.dumps(result) if result else None, datetime.now(), 
                   datetime.now() if status == 'completed' else None))
        conn.commit()
        conn.close()

    def log_metric(self, agent_id: str, metric_type: str, value: float):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO metrics (agent_id, metric_type, value, timestamp)
                     VALUES (?, ?, ?, ?)''',
                  (agent_id, metric_type, value, datetime.now()))
        conn.commit()
        conn.close()
