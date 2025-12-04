"""Tool wrappers for agent capabilities to be used by LLM."""

import logging
from typing import Dict, Any
from agents.base_agent import AgentContext

logger = logging.getLogger(__name__)


class AgentToolkit:
    """Exposes agent capabilities as callable tools."""
    
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
    
    async def git_ops(self, operation: str, arguments: list = None) -> Dict[str, Any]:
        """Execute git operations."""
        if not arguments:
            arguments = []
        
        task = {"operation": operation, "arguments": arguments}
        context = AgentContext(
            agent_id="git_agent_001",
            current_time=0,
            system_load=0,
            available_resources={},
            other_agents=[],
            active_workflows=[],
            incentive_pool=0
        )
        
        result = await self.agents["git_agent_001"].execute_task(task, context)
        return {
            "success": result.success,
            "data": result.result_data,
            "error": result.error_message
        }
    
    async def run_test(self, command: str, target_file: str = None) -> Dict[str, Any]:
        """Execute test suite."""
        task = {"command": command, "target_file": target_file}
        context = AgentContext(
            agent_id="test_agent_001",
            current_time=0,
            system_load=0,
            available_resources={},
            other_agents=[],
            active_workflows=[],
            incentive_pool=0
        )
        
        result = await self.agents["test_agent_001"].execute_task(task, context)
        return {
            "success": result.success,
            "data": result.result_data,
            "error": result.error_message
        }
    
    async def code_analysis(self, tool: str, path: str) -> Dict[str, Any]:
        """Run code analysis."""
        task = {"tool": tool, "path": path}
        context = AgentContext(
            agent_id="analysis_agent_001",
            current_time=0,
            system_load=0,
            available_resources={},
            other_agents=[],
            active_workflows=[],
            incentive_pool=0
        )
        
        result = await self.agents["analysis_agent_001"].execute_task(task, context)
        return {
            "success": result.success,
            "data": result.result_data,
            "error": result.error_message
        }
    
    async def knowledge_manager(self, action: str, key: str, content: str = None) -> Dict[str, Any]:
        """Manage semantic memory."""
        task = {"action": action, "key": key, "content": content}
        context = AgentContext(
            agent_id="knowledge_agent_001",
            current_time=0,
            system_load=0,
            available_resources={},
            other_agents=[],
            active_workflows=[],
            incentive_pool=0
        )
        
        result = await self.agents["knowledge_agent_001"].execute_task(task, context)
        return {
            "success": result.success,
            "data": result.result_data,
            "error": result.error_message
        }
