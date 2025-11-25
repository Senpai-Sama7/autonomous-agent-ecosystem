"""
Natural Language Interface for Autonomous Agent Ecosystem
"""
import logging
import json
import uuid
from typing import Dict, Any, List, Optional
from core.engine import AgentEngine, Workflow, Task, WorkflowPriority

logger = logging.getLogger("NaturalLanguageInterface")

class NaturalLanguageInterface:
    """
    Translates natural language user requests into structured workflows and tasks.
    Uses an LLM to parse intent and extract parameters.
    """
    
    def __init__(self, engine: AgentEngine, llm_client: Any = None):
        self.engine = engine
        self.llm_client = llm_client
        
    async def process_request(self, user_input: str) -> str:
        """
        Process a natural language request and submit a workflow.
        Returns the workflow ID.
        """
        logger.info(f"Processing NL request: {user_input}")
        
        if not self.llm_client:
            # Fallback for when no LLM is configured - simple keyword matching
            return await self._process_keyword_request(user_input)
            
        try:
            # Use LLM to parse request
            workflow_plan = await self._parse_intent_with_llm(user_input)
            return await self._submit_parsed_workflow(workflow_plan)
        except Exception as e:
            logger.error(f"Failed to process NL request: {e}")
            raise

    async def _parse_intent_with_llm(self, user_input: str) -> Dict[str, Any]:
        """Use LLM to convert text to structured workflow definition"""
        system_prompt = """
        You are an AI Agent Orchestrator. Your job is to convert user requests into a JSON workflow definition.
        
        Available Agents & Capabilities:
        1. Research Agent (capabilities: web_search, content_extraction, knowledge_synthesis)
        2. Code Agent (capabilities: code_generation, code_optimization, debugging)
        3. FileSystem Agent (capabilities: file_operations, data_processing)
        
        Output Format (JSON only):
        {
            "name": "Workflow Name",
            "tasks": [
                {
                    "description": "Task description",
                    "required_capabilities": ["capability1", "capability2"],
                    "payload": { ... specific params based on agent type ... }
                }
            ]
        }
        
        Payload Schemas:
        - Research: {"query": "search term", "research_type": "web_search"|"content_extraction"}
        - Code: {"code_task_type": "generate_code"|"execute_code", "requirements": "..."}
        - FileSystem: {"operation": "write_file"|"read_file"|"list_dir", "path": "...", "content": "..."}
        """
        
        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo", # Or configured model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)

    async def _submit_parsed_workflow(self, plan: Dict[str, Any]) -> str:
        """Convert parsed plan into actual Workflow object and submit"""
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        tasks = []
        for i, task_data in enumerate(plan.get("tasks", [])):
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            
            # Simple dependency: assume sequential execution for now
            dependencies = [tasks[-1].task_id] if i > 0 else []
            
            task = Task(
                task_id=task_id,
                description=task_data.get("description", "Unknown task"),
                required_capabilities=task_data.get("required_capabilities", []),
                priority=WorkflowPriority.MEDIUM,
                dependencies=dependencies,
                payload=task_data.get("payload", {})
            )
            tasks.append(task)
            
        workflow = Workflow(
            workflow_id=workflow_id,
            name=plan.get("name", "Generated Workflow"),
            tasks=tasks,
            priority=WorkflowPriority.MEDIUM
        )
        
        await self.engine.submit_workflow(workflow)
        logger.info(f"Created workflow {workflow_id} from NL request")
        return workflow_id

    async def _process_keyword_request(self, user_input: str) -> str:
        """Fallback: Simple keyword-based workflow generation"""
        user_input = user_input.lower()
        
        tasks = []
        if "research" in user_input or "search" in user_input:
            tasks.append(Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description=f"Research request: {user_input}",
                required_capabilities=["web_search"],
                payload={"query": user_input, "research_type": "web_search"}
            ))
            
        if "code" in user_input or "script" in user_input:
            tasks.append(Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description=f"Code request: {user_input}",
                required_capabilities=["code_generation"],
                dependencies=[t.task_id for t in tasks], # Depend on previous
                payload={"code_task_type": "generate_code", "requirements": user_input}
            ))
            
        if not tasks:
            raise ValueError("Could not determine intent from keywords. Please enable LLM for full NL support.")
            
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        workflow = Workflow(
            workflow_id=workflow_id,
            name=f"Keyword Workflow: {user_input[:20]}...",
            tasks=tasks
        )
        
        await self.engine.submit_workflow(workflow)
        return workflow_id
