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
import logging
import os
import tempfile
import re
import shutil
import sys
import ast
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentCapability, AgentContext, TaskResult, AgentState
from core.llm_factory import LLMFactory

# Try to import openai with async support
try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None
    AsyncOpenAI = None

# Check if Docker is available for sandboxed execution
HAS_DOCKER = shutil.which('docker') is not None

logger = logging.getLogger("CodeAgent")

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
            '--network', 'none',             # MANDATORY: No network access
            '--read-only',                   # MANDATORY: Read-only filesystem
            '--cap-drop', 'ALL',             # MANDATORY: Drop all capabilities
            '--security-opt', 'no-new-privileges',  # MANDATORY: No privilege escalation
        ]
        
        temp_path = None
        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            # Make temp file readable (skip on Windows where chmod behaves differently)
            if sys.platform != 'win32':
                os.chmod(temp_path, 0o644)
            
            # Build Docker command - security flags are ALWAYS applied first
            docker_cmd = ['docker', 'run', '--rm']
            
            # SECURITY: Add hardcoded security flags (non-overridable)
            docker_cmd.extend(HARDCODED_SECURITY_FLAGS)
            
            # Add configurable resource limits (safe to customize)
            docker_cmd.extend([
                '--memory', '128m',              # Memory limit
                '--cpus', '0.5',                 # CPU limit
            ])
            
            # Add volume mount and working directory
            docker_cmd.extend([
                '-v', f'{temp_path}:/code/script.py:ro',  # Mount code read-only
                '-w', '/code',                   # Working directory
                self.docker_image,
                'python', '/code/script.py'
            ])
            
            # SECURITY: Log that we're using enforced isolation
            logger.info(f"Executing code in Docker sandbox ({self.docker_image}) with enforced network isolation")
            
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.docker_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return TaskResult(success=False, error_message=f"Docker execution timed out ({self.docker_timeout}s limit)")
            
            if proc.returncode == 0:
                return TaskResult(
                    success=True, 
                    result_data={
                        'output': stdout.decode('utf-8', errors='replace'),
                        'execution_environment': 'docker',
                        'docker_image': self.docker_image
                    }
                )
            else:
                return TaskResult(
                    success=False, 
                    error_message=f"Docker Execution Error: {stderr.decode('utf-8', errors='replace')}"
                )
                
        except FileNotFoundError:
            return TaskResult(success=False, error_message="Docker not found. Please install Docker or disable sandbox mode.")
        except Exception as e:
            return TaskResult(success=False, error_message=f"Docker execution failed: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
    
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
            r'__import__\s*\(',           # Dynamic imports
            r'\bgetattr\s*\([^)]*__',     # getattr with dunder access
            r'\bsetattr\s*\([^)]*__',     # setattr with dunder access
            r'\bdelattr\s*\([^)]*__',     # delattr with dunder access
            r'globals\s*\(\s*\)',         # globals() access
            r'locals\s*\(\s*\)',          # locals() access
            r'\bopen\s*\([^)]*["\'][wax]', # Write mode file operations
            r'base64\s*\.\s*b64decode',   # Base64 decoding (common obfuscation)
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return TaskResult(
                    success=False, 
                    error_message="Security Alert: Code contains potentially dangerous pattern. Use Docker sandbox for untrusted code."
                )
        
        return None
    
    def _ast_security_check(self, code: str) -> Optional[TaskResult]:
        """
        AST-based security validation - analyzes code structure, not strings.
        
        This catches dangerous patterns that regex would miss due to whitespace,
        comments, or string formatting variations.
        """
        # Dangerous built-in functions
        DANGEROUS_CALLS = {'exec', 'eval', 'compile', '__import__', 'breakpoint'}
        
        # Dangerous modules
        DANGEROUS_MODULES = {
            'os', 'sys', 'subprocess', 'shutil', 'socket', 'pickle', 
            'shelve', 'marshal', 'ctypes', 'multiprocessing', 'pty',
            'commands', 'popen2', 'importlib'
        }
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return TaskResult(
                success=False,
                error_message=f"Syntax error in code: {e}"
            )
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in DANGEROUS_CALLS:
                        return TaskResult(
                            success=False,
                            error_message=f"Security Alert: Blocked dangerous call to '{node.func.id}()'. Use Docker sandbox for untrusted code."
                        )
                # Check for getattr(__builtins__, ...) pattern
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in DANGEROUS_CALLS:
                        return TaskResult(
                            success=False,
                            error_message=f"Security Alert: Blocked dangerous call to '{node.func.attr}()'. Use Docker sandbox."
                        )
            
            # Check for dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in DANGEROUS_MODULES:
                        return TaskResult(
                            success=False,
                            error_message=f"Security Alert: Import of '{module_name}' blocked. Use Docker sandbox for system access."
                        )
            
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in DANGEROUS_MODULES:
                        return TaskResult(
                            success=False,
                            error_message=f"Security Alert: Import from '{module_name}' blocked. Use Docker sandbox for system access."
                        )
        
        return None