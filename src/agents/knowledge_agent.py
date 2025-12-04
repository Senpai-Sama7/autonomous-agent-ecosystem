"""Knowledge manager agent for semantic memory persistence."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from .base_agent import BaseAgent, AgentCapability, AgentContext, TaskResult

logger = logging.getLogger(__name__)


class KnowledgeAgent(BaseAgent):
    """Manages persistent semantic memory for architectural decisions and context."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, [AgentCapability.DATA_PROCESSING], config)
        self.knowledge_dir = Path(config.get("knowledge_dir", "./workspace/knowledge"))
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    async def execute_task(self, task: Dict[str, Any], context: AgentContext) -> TaskResult:
        """Save or retrieve knowledge."""
        try:
            action = task.get("action")
            key = task.get("key")
            content = task.get("content")
            
            if action == "save":
                if not key or not content:
                    return TaskResult(success=False, error_message="Key and content required for save")
                
                file_path = self.knowledge_dir / f"{key}.json"
                with open(file_path, "w") as f:
                    json.dump({"key": key, "content": content}, f, indent=2)
                
                return TaskResult(
                    success=True,
                    result_data={"action": "save", "key": key, "path": str(file_path)}
                )
            
            elif action == "retrieve":
                if not key:
                    return TaskResult(success=False, error_message="Key required for retrieve")
                
                file_path = self.knowledge_dir / f"{key}.json"
                if not file_path.exists():
                    return TaskResult(success=False, error_message=f"Knowledge not found: {key}")
                
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                return TaskResult(
                    success=True,
                    result_data={"action": "retrieve", "key": key, "content": data.get("content")}
                )
            
            else:
                return TaskResult(success=False, error_message=f"Unknown action: {action}")
        
        except Exception as e:
            logger.error(f"Knowledge operation failed: {e}")
            return TaskResult(success=False, error_message=str(e))
