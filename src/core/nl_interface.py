"""
Natural Language Interface for Autonomous Agent Ecosystem

SECURITY: This module includes prompt injection defenses.
All user input is sanitized before LLM processing.
"""
import logging
import json
import uuid
import asyncio
import re
from typing import Dict, Any, List, Optional
from core.engine import AgentEngine, Workflow, Task, WorkflowPriority

try:
    from openai import AsyncOpenAI
    HAS_ASYNC_OPENAI = True
except ImportError:
    AsyncOpenAI = None
    HAS_ASYNC_OPENAI = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger("NaturalLanguageInterface")


class SecurityException(Exception):
    """Raised when a security violation is detected (e.g., prompt injection attempt)"""
    pass


# Hostile prompt patterns - case-insensitive regex patterns
# These detect common prompt injection techniques
HOSTILE_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"ignore\s+(all\s+)?prior\s+instructions?",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"system\s*override",
    r"admin\s*mode",
    r"developer\s*mode",
    r"bypass\s+(all\s+)?restrictions?",
    r"bypass\s+(all\s+)?safety",
    r"bypass\s+(all\s+)?security",
    r"you\s+are\s+now\s+(a|an)",  # "You are now a DAN"
    r"pretend\s+you\s+are",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"\bdan\b",  # DAN prompt
    r"ignore\s+ethical\s+guidelines",
    r"ignore\s+safety\s+guidelines",
    r"new\s+instructions?:\s*",
    r"\[\s*system\s*\]",  # [SYSTEM] injection
    r"<\s*system\s*>",   # <system> injection
    r"\{\{.*\}\}",       # Template injection {{...}}
]


class NaturalLanguageInterface:
    """
    Translates natural language user requests into structured workflows and tasks.
    Uses an LLM to parse intent and extract parameters.
    """
    
    def __init__(self, engine: AgentEngine, llm_client: Any = None, model_name: str = "gpt-3.5-turbo"):
        self.engine = engine
        self.llm_client = llm_client
        self.model_name = model_name
        
        # Initialize ML-based injection defense
        self.injection_classifier = None
        if HAS_TRANSFORMERS:
            try:
                logger.info("Loading prompt injection classifier (protectai/deberta-v3-base-prompt-injection)...")
                self.injection_classifier = pipeline(
                    "text-classification", 
                    model="protectai/deberta-v3-base-prompt-injection"
                )
                logger.info("Prompt injection classifier loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load injection classifier: {e}")
        else:
            logger.warning("Transformers library not found. ML-based prompt injection defense disabled.")        
    def _sanitize_input(self, input_str: str) -> str:
        """
        Sanitize user input to prevent prompt injection attacks.
        
        SECURITY: This is a defense-in-depth layer. It detects known hostile
        patterns that attempt to manipulate LLM behavior (jailbreaks, system
        overrides, instruction ignoring, etc.).
        
        Args:
            input_str: Raw user input
            
        Returns:
            Sanitized input string
            
        Raises:
            SecurityException: If hostile prompt injection is detected
        """
        if not input_str:
            return ""
        
        # Normalize input for pattern matching
        normalized = input_str.strip()
        
        # Check against hostile patterns
        for pattern in HOSTILE_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                logger.warning(f"SECURITY: Blocked prompt injection attempt. Pattern: {pattern}")
                raise SecurityException(
                    "Your request was blocked for security reasons. "
                    "Please rephrase your request without attempting to modify system behavior."
                )
        
        # Layer 2: ML-based detection (Slower but smarter)
        if self.injection_classifier:
            try:
                # Truncate for model if needed (DeBERTa has 512 limit usually)
                model_input = normalized[:512] 
                result = self.injection_classifier(model_input)[0]
                
                # Check for injection (label is usually 'INJECTION' or 'SAFE')
                if result['label'] == 'INJECTION' and result['score'] > 0.9:
                    logger.warning(f"SECURITY: ML model detected prompt injection (score: {result['score']:.2f})")
                    raise SecurityException("Adversarial input detected by AI defense system")
            except SecurityException:
                raise
            except Exception as e:
                logger.error(f"Error during ML injection check: {e}")
                # Fail open for availability, but log error
        
        # Length limit to prevent token exhaustion attacks
        max_length = 10000
        if len(normalized) > max_length:
            logger.warning(f"SECURITY: Input truncated from {len(normalized)} to {max_length} chars")
            normalized = normalized[:max_length]
        
        # Strip potential control characters (except newlines/tabs)
        normalized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', normalized)
        
        return normalized
    
    async def process_request(self, user_input: str) -> str:
        """
        Process a natural language request and submit a workflow.
        Returns the workflow ID.
        
        SECURITY: Input is sanitized before any LLM interaction.
        """
        # SECURITY: Sanitize input FIRST, before any processing
        try:
            sanitized_input = self._sanitize_input(user_input)
        except SecurityException as e:
            logger.warning(f"Security exception during input sanitization: {e}")
            return f"BLOCKED: {e}"
        
        logger.info(f"Processing NL request: {sanitized_input[:100]}...")
        
        if not self.llm_client:
            # Fallback for when no LLM is configured - simple keyword matching
            return await self._process_keyword_request(sanitized_input)
            
        try:
            # Use LLM to parse request (with sanitized input)
            workflow_plan = await self._parse_intent_with_llm(sanitized_input)
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
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        # Check if client is async
        is_async = HAS_ASYNC_OPENAI and isinstance(self.llm_client, AsyncOpenAI)
        
        if is_async:
            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
        else:
            # Fallback to sync call (wrapped in thread if needed, but here we assume sync client is ok)
            # Ideally, we should run sync client in a thread to avoid blocking
            if hasattr(self.llm_client.chat.completions, 'create'):
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
            else:
                raise ValueError("Invalid LLM client provided")
        
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
