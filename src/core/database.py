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

# CLI interface
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Database Management CLI')
    parser.add_argument('command', choices=['backup', 'restore'], help='Command to execute')
    parser.add_argument('--db', default='ecosystem.db', help='Database file path')
    parser.add_argument('--file', required=True, help='Backup file path')
    
    args = parser.parse_args()
    
    db = DatabaseManager(args.db)
    
    if args.command == 'backup':
        if db.backup(args.file):
            print(f"✅ Backup created: {args.file}")
            sys.exit(0)
        else:
            print(f"❌ Backup failed")
            sys.exit(1)
    elif args.command == 'restore':
        if db.restore(args.file):
            print(f"✅ Database restored from: {args.file}")
            sys.exit(0)
        else:
            print(f"❌ Restore failed")
            sys.exit(1)
