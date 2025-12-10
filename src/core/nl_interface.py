"""
Natural Language Interface for Autonomous Agent Ecosystem

SECURITY: This module includes prompt injection defenses.
All user input is sanitized before LLM processing.

ENHANCED: Integrates Zero Reasoning for structured reasoning
on complex requests, improving intent extraction accuracy.
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

# Zero Reasoning integration for structured intent extraction
try:
    from core.zero_reasoning import create_reasoner, AbsoluteZeroReasoner, ReasoningMode
    HAS_ZERO_REASONING = True
except ImportError:
    AbsoluteZeroReasoner = None
    HAS_ZERO_REASONING = False

logger = logging.getLogger("NaturalLanguageInterface")


class SecurityException(Exception):
    """Raised when a security violation is detected (e.g., prompt injection attempt)"""
    pass


# Hostile prompt patterns - case-insensitive regex patterns
# These detect common prompt injection techniques
HOSTILE_PATTERNS = [
    # Instruction override attempts
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"ignore\s+(all\s+)?prior\s+instructions?",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"override\s+(all\s+)?instructions?",
    r"new\s+instructions?:\s*",
    
    # Mode switching attempts
    r"system\s*override",
    r"admin\s*mode",
    r"developer\s*mode",
    r"debug\s*mode",
    r"maintenance\s*mode",
    r"god\s*mode",
    r"unrestricted\s*mode",
    
    # Bypass attempts
    r"bypass\s+(all\s+)?restrictions?",
    r"bypass\s+(all\s+)?safety",
    r"bypass\s+(all\s+)?security",
    r"bypass\s+(all\s+)?filters?",
    r"bypass\s+(all\s+)?guidelines",
    r"remove\s+(all\s+)?restrictions?",
    r"disable\s+(all\s+)?restrictions?",
    
    # Role-playing jailbreaks
    r"you\s+are\s+now\s+(a|an)",  # "You are now a DAN"
    r"pretend\s+you\s+are",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
    r"roleplay\s+as",
    r"imagine\s+you\s+are\s+a\s+different",
    
    # Known jailbreak patterns
    r"jailbreak",
    r"do\s+anything\s+now",
    r"\bdan\b",  # DAN prompt
    r"\bdeveloper\s*mode\s*enabled\b",
    r"stay\s+in\s+character",
    
    # Guideline bypass
    r"ignore\s+ethical\s+guidelines",
    r"ignore\s+safety\s+guidelines",
    r"ignore\s+your\s+(rules|guidelines|constraints)",
    r"don'?t\s+follow\s+your\s+(rules|guidelines)",
    
    # Injection markers
    r"\[\s*system\s*\]",  # [SYSTEM] injection
    r"<\s*system\s*>",   # <system> injection
    r"\{\{.*\}\}",       # Template injection {{...}}
    r"<\|.*\|>",         # Special token injection
    r"###\s*instruction",  # Markdown-style injection
    
    # Prompt leaking attempts
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"show\s+(me\s+)?(your\s+)?instructions",
    r"what\s+are\s+your\s+instructions",
    r"repeat\s+(your\s+)?system\s+prompt",
]

# Additional heuristic checks when ML classifier is not available
INJECTION_HEURISTICS = {
    "instruction_ratio": 0.15,       # Max ratio of instruction-like words
    "special_char_ratio": 0.10,      # Max ratio of special characters
    "repeated_pattern_threshold": 3,  # Max times a suspicious phrase can repeat
}

# Words commonly used in injection attempts
INJECTION_KEYWORDS = {
    "ignore", "override", "bypass", "disable", "forget", "disregard",
    "pretend", "roleplay", "jailbreak", "unrestricted", "uncensored",
    "admin", "system", "developer", "instruction", "prompt", "constraint"
}


class NaturalLanguageInterface:
    """
    Translates natural language user requests into structured workflows and tasks.
    Uses an LLM to parse intent and extract parameters.

    Enhanced with Zero Reasoning for complex multi-step request analysis.
    """

    def __init__(self, engine: AgentEngine, llm_client: Any = None, model_name: str = "gpt-3.5-turbo",
                 enable_reasoning: bool = True):
        self.engine = engine
        self.llm_client = llm_client
        self.model_name = model_name
        self.enable_reasoning = enable_reasoning

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

        # Initialize Zero Reasoner for complex request analysis
        self.reasoner: Optional[AbsoluteZeroReasoner] = None
        if enable_reasoning and HAS_ZERO_REASONING and llm_client:
            try:
                self.reasoner = create_reasoner(llm_client, model_name)
                logger.info("Zero Reasoning engine initialized for NL processing")
            except Exception as e:
                logger.warning(f"Failed to initialize Zero Reasoner: {e}")        
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
    
    def _is_complex_request(self, user_input: str) -> bool:
        """Determine if the request requires structured reasoning analysis."""
        complex_indicators = [
            "compare", "analyze", "multiple", "alternatives", "best approach",
            "trade-offs", "pros and cons", "evaluate", "decide between",
            "which is better", "should i use", "recommend", "optimize"
        ]
        input_lower = user_input.lower()
        # Complex if input is long or contains complexity indicators
        return (
            len(user_input) > 200 or
            any(indicator in input_lower for indicator in complex_indicators)
        )

    async def _analyze_with_reasoning(self, user_input: str) -> Dict[str, Any]:
        """
        Use Zero Reasoning for structured analysis of complex requests.
        Returns enhanced context for workflow generation.
        """
        if not self.reasoner:
            return {"enhanced": False}

        try:
            logger.info("Using Zero Reasoning for complex request analysis...")

            # Perform chain-of-thought reasoning
            result = await self.reasoner.reason(
                question=f"What tasks and capabilities are needed to fulfill this user request? Request: {user_input}",
                context="Available agents: Research (web_search, content_extraction), Code (code_generation, debugging), FileSystem (file_operations)",
                mode=ReasoningMode.DEDUCTIVE
            )

            return {
                "enhanced": True,
                "reasoning_answer": result.get("answer", ""),
                "confidence": result.get("confidence", 0.0),
                "reasoning_steps": result.get("steps", []),
                "reasoning_type": result.get("reasoning_type", "chain_of_thought")
            }
        except Exception as e:
            logger.warning(f"Zero Reasoning analysis failed: {e}")
            return {"enhanced": False, "error": str(e)}

    async def process_request(self, user_input: str) -> str:
        """
        Process a natural language request and submit a workflow.
        Returns the workflow ID.

        SECURITY: Input is sanitized before any LLM interaction.
        ENHANCED: Uses Zero Reasoning for complex requests.
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
            # Check if request is complex and needs reasoning analysis
            reasoning_context = {}
            if self.enable_reasoning and self.reasoner and self._is_complex_request(sanitized_input):
                reasoning_context = await self._analyze_with_reasoning(sanitized_input)
                if reasoning_context.get("enhanced"):
                    logger.info(f"Reasoning analysis complete (confidence: {reasoning_context.get('confidence', 0):.2f})")

            # Use LLM to parse request (with sanitized input and optional reasoning context)
            workflow_plan = await self._parse_intent_with_llm(sanitized_input, reasoning_context)
            return await self._submit_parsed_workflow(workflow_plan)
        except Exception as e:
            logger.error(f"Failed to process NL request: {e}")
            raise

    async def _parse_intent_with_llm(self, user_input: str, reasoning_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Use LLM to convert text to structured workflow definition.

        Args:
            user_input: Sanitized user request
            reasoning_context: Optional context from Zero Reasoning analysis
        """
        # Build enhanced context from reasoning if available
        reasoning_guidance = ""
        if reasoning_context and reasoning_context.get("enhanced"):
            reasoning_guidance = f"""

        REASONING ANALYSIS (use this to inform your task breakdown):
        Analysis: {reasoning_context.get('reasoning_answer', '')}
        Confidence: {reasoning_context.get('confidence', 0.0):.2f}
        """

        system_prompt = f"""
        You are an AI Agent Orchestrator. Your job is to convert user requests into a JSON workflow definition.

        Available Agents & Capabilities:
        1. Research Agent (capabilities: web_search, content_extraction, knowledge_synthesis)
        2. Code Agent (capabilities: code_generation, code_optimization, debugging)
        3. FileSystem Agent (capabilities: file_operations, data_processing)

        Output Format (JSON only):
        {{
            "name": "Workflow Name",
            "tasks": [
                {{
                    "description": "Task description",
                    "required_capabilities": ["capability1", "capability2"],
                    "payload": {{ ... specific params based on agent type ... }}
                }}
            ]
        }}

        Payload Schemas:
        - Research: {{"query": "search term", "research_type": "web_search"|"content_extraction"}}
        - Code: {{"code_task_type": "generate_code"|"execute_code", "requirements": "..."}}
        - FileSystem: {{"operation": "write_file"|"read_file"|"list_dir", "path": "...", "content": "..."}}
        {reasoning_guidance}
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
