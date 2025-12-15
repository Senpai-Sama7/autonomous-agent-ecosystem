"""
Code Agent with Real Generation and Execution Capabilities

REFACTORED: Full async implementation.
- Async subprocess execution (asyncio.create_subprocess_exec)
- Async LLM API calls (AsyncOpenAI)
- Docker sandbox REQUIRED for code execution in production
- Local execution disabled by default (security risk)

SECURITY WARNING:
    Regex-based blocklisting is fundamentally insecure. Attackers can bypass
    string matching via getattr(__import__('os'), 'system'), base64 encoding,
    or other dynamic execution paths. Always use Docker sandbox in production.
"""

import asyncio
import os
import tempfile
import re
import shutil
import sys
import ast
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentCapability, AgentContext, TaskResult, AgentState
from core.llm_factory import LLMFactory

# Structured logging
from src.utils.structured_logger import get_logger, log_performance

# Try to import openai with async support
try:
    from openai import OpenAI, AsyncOpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None
    AsyncOpenAI = None

# Check if Docker is available for sandboxed execution
HAS_DOCKER = shutil.which("docker") is not None

logger = get_logger("CodeAgent")


class CodeAgent(BaseAgent):
    """
    Code generation and execution agent with security-first design.

    SECURITY ARCHITECTURE:
    - Docker sandbox is MANDATORY for code execution in production
    - Local execution requires explicit opt-in AND Docker unavailability
    - AST + regex validation provides defense-in-depth but is NOT foolproof
    - All security flags are hardcoded and cannot be overridden by config
    """

    def __init__(self, agent_id: str, config: Dict[str, Any], llm_client: Any = None):
        """
        Initialize CodeAgent with security-first defaults.

        Args:
            agent_id: Unique identifier for this agent
            config: Configuration dict from agents.yaml
            llm_client: Optional pre-configured LLM client (AsyncOpenAI)
        """
        capabilities = [AgentCapability.DATA_PROCESSING, AgentCapability.OPTIMIZATION]
        super().__init__(agent_id, capabilities, config)

        # LLM Configuration
        self.llm_client = llm_client
        self.model_name = config.get("model_name", "gpt-3.5-turbo")

        # Security Configuration - HARDENED for production
        # safe_mode is ALWAYS True - cannot be disabled via config
        self.safe_mode = True  # HARDCODED - ignore config
        self.use_docker_sandbox = True  # HARDCODED - always require Docker
        self.allow_local_execution = bool(config.get("allow_local_execution", False))
        self.docker_image = config.get("docker_image", "python:3.11-slim")
        self.docker_timeout = config.get("docker_execution_timeout", 30)
        self._docker_available = HAS_DOCKER

        self._validate_execution_environment()

        logger.info(
            f"CodeAgent {agent_id} initialized (Docker: {HAS_DOCKER}, Sandbox: ENFORCED)"
        )

    def _validate_execution_environment(self) -> None:
        """
        Validate execution environment at startup and emit warnings.

        This provides early warning if the system is misconfigured.
        """
        if self.use_docker_sandbox and not self._docker_available:
            logger.warning(
                "SECURITY WARNING: Docker sandbox is enabled but Docker is not installed. "
                "Code execution will be DISABLED unless allow_local_execution is True."
            )

        if self.allow_local_execution and not self.use_docker_sandbox:
            logger.warning(
                "SECURITY WARNING: Local code execution is enabled WITHOUT Docker sandbox. "
                "This is a security risk. Only enable for trusted code in development environments."
            )

        if not self.safe_mode:
            logger.warning(
                "SECURITY WARNING: safe_mode is disabled. "
                "AST/regex validation will not be performed on generated code."
            )

    async def execute_task(
        self, task: Dict[str, Any], context: AgentContext
    ) -> TaskResult:
        """
        Execute a code-related task.

        Supported task types:
        - generate_code: Generate Python code using LLM
        - execute_code: Execute code in sandbox
        - analyze_code: Analyze code for issues
        """
        self.state = AgentState.BUSY
        task_type = task.get("code_task_type", "generate_code")

        try:
            if task_type == "generate_code":
                return await self._generate_code(task)
            elif task_type == "execute_code":
                return await self._execute_code(task.get("code", ""))
            elif task_type == "analyze_code":
                return await self._analyze_code(task.get("code", ""))
            else:
                return TaskResult(
                    success=False, error_message=f"Unknown task type: {task_type}"
                )
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return TaskResult(success=False, error_message=str(e))
        finally:
            self.state = AgentState.IDLE

    async def _generate_code(self, task: Dict[str, Any]) -> TaskResult:
        """Generate Python code using LLM."""
        if not self.llm_client:
            return TaskResult(
                success=False,
                error_message="LLM client not configured. Cannot generate code.",
            )

        requirements = task.get("requirements", "")
        if not requirements:
            return TaskResult(
                success=False,
                error_message="No requirements provided for code generation.",
            )

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Python code generator. Generate clean, safe, well-documented code. "
                            "Do NOT use os, sys, subprocess, or any system-level modules. "
                            "Return ONLY the Python code, no explanations or markdown."
                        ),
                    },
                    {"role": "user", "content": requirements},
                ],
                max_tokens=2000,
                temperature=0.3,
            )

            code = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()

            # Security check on generated code
            security_result = self._security_check(code)
            if security_result:
                return security_result

            return TaskResult(
                success=True, result_data={"code": code, "model": self.model_name}
            )

        except Exception as e:
            return TaskResult(
                success=False, error_message=f"Code generation failed: {e}"
            )

    async def _execute_code(self, code: str) -> TaskResult:
        """
        Execute code in the safest available environment.

        Execution priority:
        1. Docker sandbox (preferred, mandatory in production)
        2. Local execution (only if explicitly enabled AND Docker unavailable)
        3. Refuse execution (if neither option available)
        """
        if not code:
            return TaskResult(
                success=False, error_message="No code provided for execution."
            )

        # Security check first
        security_result = self._security_check(code)
        if security_result:
            return security_result

        # Determine execution environment
        if self.use_docker_sandbox:
            if self._docker_available:
                return await self._execute_in_docker(code)
            elif self.allow_local_execution:
                logger.warning(
                    "Docker unavailable, falling back to local execution (SECURITY RISK)"
                )
                return await self._execute_locally(code)
            else:
                return TaskResult(
                    success=False,
                    error_message=(
                        "Docker sandbox is required but Docker is not installed. "
                        "Install Docker or set allow_local_execution=true in config (NOT RECOMMENDED)."
                    ),
                )
        elif self.allow_local_execution:
            return await self._execute_locally(code)
        else:
            return TaskResult(
                success=False,
                error_message="Code execution is disabled. Enable Docker sandbox or allow_local_execution.",
            )

    async def _execute_in_docker(self, code: str) -> TaskResult:
        """
        Execute code in a Docker sandbox with hardcoded security restrictions.

        SECURITY: Container is completely isolated with NO network access.
        This is hardcoded and CANNOT be overridden by configuration.

        Container features:
        - No network access (--network none) - HARDCODED, NON-NEGOTIABLE
        - Read-only filesystem (--read-only)
        - Limited memory (--memory 128m)
        - Limited CPU (--cpus 0.5)
        - Dropped capabilities (--cap-drop ALL)
        - No new privileges (--security-opt no-new-privileges)
        - Auto-removed after execution (--rm)
        """
        # SECURITY: These flags are HARDCODED and cannot be overridden
        # Even if config attempts to pass network arguments, they are ignored
        HARDCODED_SECURITY_FLAGS = [
            "--network",
            "none",  # MANDATORY: No network access
            "--read-only",  # MANDATORY: Read-only filesystem
            "--cap-drop",
            "ALL",  # MANDATORY: Drop all capabilities
            "--security-opt",
            "no-new-privileges",  # MANDATORY: No privilege escalation
        ]

        temp_path = None
        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_path = f.name

            # Make temp file readable (skip on Windows where chmod behaves differently)
            if sys.platform != "win32":
                os.chmod(temp_path, 0o644)

            # Build Docker command - security flags are ALWAYS applied first
            docker_cmd = ["docker", "run", "--rm"]

            # SECURITY: Add hardcoded security flags (non-overridable)
            docker_cmd.extend(HARDCODED_SECURITY_FLAGS)

            # Add configurable resource limits (safe to customize)
            docker_cmd.extend(
                [
                    "--memory",
                    "128m",  # Memory limit
                    "--cpus",
                    "0.5",  # CPU limit
                ]
            )

            # Add volume mount and working directory
            docker_cmd.extend(
                [
                    "-v",
                    f"{temp_path}:/code/script.py:ro",  # Mount code read-only
                    "-w",
                    "/code",  # Working directory
                    self.docker_image,
                    "python",
                    "/code/script.py",
                ]
            )

            # SECURITY: Log that we're using enforced isolation
            logger.info(
                f"Executing code in Docker sandbox ({self.docker_image}) with enforced network isolation"
            )

            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.docker_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return TaskResult(
                    success=False,
                    error_message=f"Docker execution timed out ({self.docker_timeout}s limit)",
                )

            if proc.returncode == 0:
                return TaskResult(
                    success=True,
                    result_data={
                        "output": stdout.decode("utf-8", errors="replace"),
                        "execution_environment": "docker",
                        "docker_image": self.docker_image,
                    },
                )
            else:
                return TaskResult(
                    success=False,
                    error_message=f"Docker Execution Error: {stderr.decode('utf-8', errors='replace')}",
                )

        except FileNotFoundError:
            msg = "Docker not found. Please install Docker or disable sandbox mode."
            return TaskResult(success=False, error_message=msg)
        except Exception as e:
            logger.error(f"CodeAgent Docker execution failed: {e}", exc_info=True)
            msg = f"Docker execution failed: {e}"
            return TaskResult(success=False, error_message=msg)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    async def _execute_locally(self, code: str) -> TaskResult:
        """
        Execute code locally (SECURITY RISK - use only in development).

        WARNING: This bypasses Docker isolation. Only use for trusted code.
        """
        logger.warning("SECURITY: Executing code locally without Docker sandbox")

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_path = f.name

            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=30,  # 30 second timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return TaskResult(
                    success=False, error_message="Local execution timed out (30s limit)"
                )

            if proc.returncode == 0:
                return TaskResult(
                    success=True,
                    result_data={
                        "output": stdout.decode("utf-8", errors="replace"),
                        "execution_environment": "local",
                    },
                )
            else:
                return TaskResult(
                    success=False,
                    error_message=f"Execution Error: {stderr.decode('utf-8', errors='replace')}",
                )

        except Exception as e:
            return TaskResult(
                success=False, error_message=f"Local execution failed: {e}"
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    async def _analyze_code(self, code: str) -> TaskResult:
        """Analyze code for potential issues using AST."""
        if not code:
            return TaskResult(
                success=False, error_message="No code provided for analysis."
            )

        try:
            tree = ast.parse(code)

            # Basic analysis
            functions = [
                node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            ]
            classes = [
                node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            ]
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            return TaskResult(
                success=True,
                result_data={
                    "functions": functions,
                    "classes": classes,
                    "imports": imports,
                    "line_count": len(code.splitlines()),
                },
            )
        except SyntaxError as e:
            return TaskResult(success=False, error_message=f"Syntax error in code: {e}")

    def _security_check(self, code: str) -> Optional[TaskResult]:
        """
        Perform security validation on code using AST analysis + regex fallback.

        CRITICAL WARNING: This provides defense-in-depth but is NOT foolproof.

        AST validation catches:
        - Direct calls to exec(), eval(), __import__(), compile()
        - Imports of dangerous modules (os, sys, subprocess, etc.)

        It CANNOT catch:
        - getattr-based dynamic execution
        - Encoded/obfuscated payloads
        - Runtime-constructed strings

        Always use Docker sandbox for untrusted code.
        """
        if not self.safe_mode:
            return None

        # Layer 1: AST-based validation (more robust than regex)
        ast_result = self._ast_security_check(code)
        if ast_result:
            return ast_result

        # Layer 2: Regex patterns for edge cases AST might miss
        forbidden_patterns = [
            r"__import__\s*\(",  # Dynamic imports
            r"\bgetattr\s*\([^)]*__",  # getattr with dunder access
            r"\bsetattr\s*\([^)]*__",  # setattr with dunder access
            r"\bdelattr\s*\([^)]*__",  # delattr with dunder access
            r"globals\s*\(\s*\)",  # globals() access
            r"locals\s*\(\s*\)",  # locals() access
            r'\bopen\s*\([^)]*["\'][wax]',  # Write mode file operations
            r"base64\s*\.\s*b64decode",  # Base64 decoding (common obfuscation)
        ]

        for pattern in forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return TaskResult(
                    success=False,
                    error_message="Security Alert: Code contains potentially dangerous pattern. Use Docker sandbox for untrusted code.",
                )

        return None

    def _ast_security_check(self, code: str) -> Optional[TaskResult]:
        """
        AST-based security validation - analyzes code structure, not strings.

        This catches dangerous patterns that regex would miss due to whitespace,
        comments, or string formatting variations.
        """
        # Dangerous built-in functions
        DANGEROUS_CALLS = {"exec", "eval", "compile", "__import__", "breakpoint"}

        # Dangerous modules
        DANGEROUS_MODULES = {
            "os",
            "sys",
            "subprocess",
            "shutil",
            "socket",
            "pickle",
            "shelve",
            "marshal",
            "ctypes",
            "multiprocessing",
            "pty",
            "commands",
            "popen2",
            "importlib",
        }

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return TaskResult(success=False, error_message=f"Syntax error in code: {e}")

        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in DANGEROUS_CALLS:
                        return TaskResult(
                            success=False,
                            error_message=f"Security Alert: Blocked dangerous call to '{node.func.id}()'. Use Docker sandbox for untrusted code.",
                        )
                # Check for getattr(__builtins__, ...) pattern
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in DANGEROUS_CALLS:
                        return TaskResult(
                            success=False,
                            error_message=f"Security Alert: Blocked dangerous call to '{node.func.attr}()'. Use Docker sandbox.",
                        )

            # Check for dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name in DANGEROUS_MODULES:
                        return TaskResult(
                            success=False,
                            error_message=f"Security Alert: Import of '{module_name}' blocked. Use Docker sandbox for system access.",
                        )

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]
                    if module_name in DANGEROUS_MODULES:
                        return TaskResult(
                            success=False,
                            error_message=f"Security Alert: Import from '{module_name}' blocked. Use Docker sandbox for system access.",
                        )

        return None
