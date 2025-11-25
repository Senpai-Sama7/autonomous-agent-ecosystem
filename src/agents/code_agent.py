"""
Code Agent with Real Generation and Execution Capabilities
"""
import asyncio
import logging
import os
import subprocess
import tempfile
import json
from typing import Dict, Any, Optional
from .base_agent import BaseAgent, AgentCapability, AgentContext, TaskResult, AgentState

# Try to import openai, handle if missing
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger("CodeAgent")

class CodeAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, [AgentCapability.OPTIMIZATION], config)
        
        self.safe_mode = config.get("safe_mode", True)
        self.provider = config.get("llm_provider", "openai").lower()
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        self.client: Optional[OpenAI] = None
        
        if HAS_OPENAI:
            self._initialize_client(config)
        else:
            logger.warning("OpenAI library not found. Code generation will be disabled.")

    def _initialize_client(self, config: Dict[str, Any]):
        """Initialize the LLM client based on provider"""
        api_key = None
        base_url = None
        
        if self.provider == "openai":
            api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
            # Default base_url
            
        elif self.provider == "openrouter":
            api_key = config.get("api_key") or os.getenv("OPENROUTER_API_KEY")
            base_url = "https://openrouter.ai/api/v1"
            if not self.model_name:
                self.model_name = "openai/gpt-3.5-turbo" # Default for OpenRouter
                
        elif self.provider == "ollama":
            api_key = "ollama" # Dummy key required by client
            base_url = config.get("api_base") or "http://localhost:11434/v1"
            if not self.model_name or self.model_name == "gpt-3.5-turbo":
                self.model_name = "llama3" # Default to llama3 if not specified
                
        else:
            logger.warning(f"Unknown provider {self.provider}, defaulting to OpenAI")
            api_key = os.getenv("OPENAI_API_KEY")

        if api_key or self.provider == "ollama":
            try:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
                logger.info(f"Initialized CodeAgent with provider: {self.provider}, model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning(f"No API key found for provider {self.provider}")

    async def execute_task(self, task: Dict[str, Any], context: AgentContext) -> TaskResult:
        try:
            self.state = AgentState.BUSY
            task_type = task.get('payload', {}).get('code_task_type', 'generate_code')
            requirements = task.get('payload', {}).get('requirements', '')
            
            if task_type == 'generate_code':
                return await self._generate_code(requirements)
            elif task_type == 'execute_code':
                code = task.get('payload', {}).get('code')
                return await self._execute_code(code)
            else:
                return TaskResult(success=False, error_message=f"Unknown task type: {task_type}")

        except Exception as e:
            logger.error(f"Code task failed: {e}")
            return TaskResult(success=False, error_message=str(e))
        finally:
            self.state = AgentState.ACTIVE

    async def _generate_code(self, requirements: str) -> TaskResult:
        """Generate code using LLM API"""
        if not self.client:
            return TaskResult(success=False, error_message=f"LLM client not initialized for {self.provider}. Cannot generate code.")

        try:
            logger.info(f"Generating code using {self.provider} ({self.model_name})...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Python code generator. Output ONLY valid Python code. No markdown backticks. Do not include any explanations."},
                    {"role": "user", "content": requirements}
                ]
            )
            code = response.choices[0].message.content
            
            # Clean up code if it contains markdown
            if code.startswith("```"):
                code = code.split("\n", 1)[1]
            if code.endswith("```"):
                code = code.rsplit("\n", 1)[0]
            if code.startswith("python"):
                code = code[6:].strip()
                
            return TaskResult(success=True, result_data={'code': code})
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return TaskResult(success=False, error_message=f"LLM Generation failed: {e}")

    async def _execute_code(self, code: str) -> TaskResult:
        """Execute Python code in a subprocess"""
        if not code:
            return TaskResult(success=False, error_message="No code provided for execution")
            
        # Security Check
        if "import os" in code or "import sys" in code or "subprocess" in code:
            if self.safe_mode:
                return TaskResult(success=False, error_message="Security Alert: Code contains forbidden imports (os, sys, subprocess). Execution blocked in Safe Mode.")

        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name

            # Execute
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=10  # 10 second timeout
            )
            
            # Cleanup
            os.unlink(temp_path)

            if result.returncode == 0:
                return TaskResult(success=True, result_data={'output': result.stdout})
            else:
                return TaskResult(success=False, error_message=f"Execution Error: {result.stderr}")

        except subprocess.TimeoutExpired:
            return TaskResult(success=False, error_message="Code execution timed out (10s limit)")
        except Exception as e:
            return TaskResult(success=False, error_message=f"Execution failed: {e}")