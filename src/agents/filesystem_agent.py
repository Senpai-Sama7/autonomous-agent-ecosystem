"""
FileSystem Agent for Safe File Operations
"""
import os
import logging
import shutil
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentCapability, AgentContext, TaskResult, AgentState

logger = logging.getLogger("FileSystemAgent")

class FileSystemAgent(BaseAgent):
    """
    Agent responsible for safe filesystem operations.
    Acts as a gatekeeper to prevent unauthorized access to system files.
    """
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, [AgentCapability.DATA_PROCESSING], config)
        self.root_dir = os.path.abspath(config.get("root_dir", "./workspace"))
        self.allowed_extensions = config.get("allowed_extensions", [".txt", ".py", ".md", ".json", ".csv", ".log"])
        
        # Ensure root dir exists
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            
    def _is_safe_path(self, path: str) -> bool:
        """Check if path is within the allowed root directory"""
        # Resolve absolute path
        abs_path = os.path.abspath(os.path.join(self.root_dir, path))
        return abs_path.startswith(self.root_dir)

    async def execute_task(self, task: Dict[str, Any], context: AgentContext) -> TaskResult:
        try:
            self.state = AgentState.BUSY
            operation = task.get('payload', {}).get('operation')
            
            if operation == 'write_file':
                return self._write_file(task.get('payload', {}))
            elif operation == 'read_file':
                return self._read_file(task.get('payload', {}))
            elif operation == 'list_dir':
                return self._list_dir(task.get('payload', {}))
            elif operation == 'make_dir':
                return self._make_dir(task.get('payload', {}))
            else:
                return TaskResult(success=False, error_message=f"Unknown operation: {operation}")

        except Exception as e:
            logger.error(f"Filesystem task failed: {e}")
            return TaskResult(success=False, error_message=str(e))
        finally:
            self.state = AgentState.ACTIVE

    def _write_file(self, payload: Dict[str, Any]) -> TaskResult:
        path = payload.get('path')
        content = payload.get('content')
        
        if not path or content is None:
            return TaskResult(success=False, error_message="Missing path or content")
            
        if not self._is_safe_path(path):
            return TaskResult(success=False, error_message=f"Access denied: Path {path} is outside workspace")
            
        full_path = os.path.join(self.root_dir, path)
        
        # Check extension
        _, ext = os.path.splitext(full_path)
        if self.allowed_extensions and ext not in self.allowed_extensions:
             return TaskResult(success=False, error_message=f"Access denied: Extension {ext} not allowed")

        try:
            # Ensure parent dir exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return TaskResult(success=True, result_data={'path': path, 'size': len(content)})
        except Exception as e:
            return TaskResult(success=False, error_message=str(e))

    def _read_file(self, payload: Dict[str, Any]) -> TaskResult:
        path = payload.get('path')
        
        if not path:
             return TaskResult(success=False, error_message="Missing path")
             
        if not self._is_safe_path(path):
            return TaskResult(success=False, error_message=f"Access denied: Path {path} is outside workspace")
            
        full_path = os.path.join(self.root_dir, path)
        
        if not os.path.exists(full_path):
            return TaskResult(success=False, error_message="File not found")
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return TaskResult(success=True, result_data={'content': content})
        except Exception as e:
            return TaskResult(success=False, error_message=str(e))

    def _list_dir(self, payload: Dict[str, Any]) -> TaskResult:
        path = payload.get('path', '.')
        
        if not self._is_safe_path(path):
            return TaskResult(success=False, error_message=f"Access denied: Path {path} is outside workspace")
            
        full_path = os.path.join(self.root_dir, path)
        
        try:
            items = os.listdir(full_path)
            return TaskResult(success=True, result_data={'items': items})
        except Exception as e:
            return TaskResult(success=False, error_message=str(e))
            
    def _make_dir(self, payload: Dict[str, Any]) -> TaskResult:
        path = payload.get('path')
        
        if not path:
             return TaskResult(success=False, error_message="Missing path")
             
        if not self._is_safe_path(path):
            return TaskResult(success=False, error_message=f"Access denied: Path {path} is outside workspace")
            
        full_path = os.path.join(self.root_dir, path)
        
        try:
            os.makedirs(full_path, exist_ok=True)
            return TaskResult(success=True, result_data={'path': path})
        except Exception as e:
            return TaskResult(success=False, error_message=str(e))
