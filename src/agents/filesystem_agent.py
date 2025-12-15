"""
FileSystem Agent for Safe File Operations
"""

import os
import shutil
import asyncio
from typing import Dict, Any, List, Optional
import aiofiles
import aiofiles.os
from .base_agent import BaseAgent, AgentCapability, AgentContext, TaskResult, AgentState

# Structured logging
from src.utils.structured_logger import get_logger, log_performance

logger = get_logger("FileSystemAgent")


class FileSystemAgent(BaseAgent):
    """
    Agent responsible for safe filesystem operations.
    Acts as a gatekeeper to prevent unauthorized access to system files.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, [AgentCapability.DATA_PROCESSING], config)
        self.root_dir = os.path.abspath(config.get("root_dir", "./workspace"))
        self.allowed_extensions = config.get(
            "allowed_extensions", [".txt", ".py", ".md", ".json", ".csv", ".log"]
        )
        self.max_file_size = config.get(
            "max_file_size", 10 * 1024 * 1024
        )  # 10MB default

        # Ensure root dir exists
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def _is_safe_path(self, path: str) -> bool:
        """
        Check if path is within the allowed root directory.

        Security: Blocks directory traversal attacks including:
        - Unix-style: ../../../etc/passwd
        - Windows-style: ..\\..\\..\\windows\\system32
        - URL-encoded: %2e%2e%2f
        - Absolute paths: /etc/passwd, C:\\Windows
        """
        # Normalize path separators (Windows -> Unix)
        normalized_path = path.replace("\\", "/")

        # Block URL-encoded traversal
        if "%2e" in path.lower() or "%2f" in path.lower():
            return False

        # Block absolute paths
        if os.path.isabs(path) or path.startswith("/"):
            return False

        # Block any path containing ..
        if ".." in normalized_path:
            return False

        # Resolve absolute path and verify it's under root
        abs_path = os.path.abspath(os.path.join(self.root_dir, normalized_path))
        try:
            return os.path.commonpath([self.root_dir, abs_path]) == self.root_dir
        except ValueError:
            return False

    async def execute_task(
        self, task: Dict[str, Any], context: AgentContext
    ) -> TaskResult:
        try:
            self.state = AgentState.BUSY
            operation = task.get("payload", {}).get("operation")

            if operation == "write_file":
                return await self._write_file(task.get("payload", {}))
            elif operation == "read_file":
                return await self._read_file(task.get("payload", {}))
            elif operation == "list_dir":
                return await self._list_dir(task.get("payload", {}))
            elif operation == "make_dir":
                return await self._make_dir(task.get("payload", {}))
            else:
                return TaskResult(
                    success=False, error_message=f"Unknown operation: {operation}"
                )

        except Exception as e:
            logger.error(f"Filesystem task failed: {e}")
            return TaskResult(success=False, error_message=str(e))
        finally:
            self.state = AgentState.ACTIVE

    async def _write_file(self, payload: Dict[str, Any]) -> TaskResult:
        path = payload.get("path")
        content = payload.get("content")

        if not path or content is None:
            return TaskResult(success=False, error_message="Missing path or content")

        # Check file size
        content_size = len(content.encode("utf-8"))
        if content_size > self.max_file_size:
            return TaskResult(
                success=False,
                error_message=f"File size ({content_size} bytes) exceeds limit ({self.max_file_size} bytes)",
            )

        if not self._is_safe_path(path):
            return TaskResult(
                success=False,
                error_message=f"Access denied: Path {path} is outside workspace",
            )

        full_path = os.path.join(self.root_dir, path)

        # Check extension
        _, ext = os.path.splitext(full_path)
        if self.allowed_extensions and ext not in self.allowed_extensions:
            return TaskResult(
                success=False,
                error_message=f"Access denied: Extension {ext} not allowed",
            )

        try:
            # Ensure parent dir exists (async)
            parent_dir = os.path.dirname(full_path)
            if not os.path.exists(parent_dir):
                await asyncio.to_thread(os.makedirs, parent_dir, 0o755, True)

            async with aiofiles.open(full_path, "w", encoding="utf-8") as f:
                await f.write(content)
            return TaskResult(
                success=True, result_data={"path": path, "size": len(content)}
            )
        except Exception as e:
            return TaskResult(success=False, error_message=str(e))

    async def _read_file(self, payload: Dict[str, Any]) -> TaskResult:
        path = payload.get("path")

        if not path:
            return TaskResult(success=False, error_message="Missing path")

        if not self._is_safe_path(path):
            return TaskResult(
                success=False,
                error_message=f"Access denied: Path {path} is outside workspace",
            )

        full_path = os.path.join(self.root_dir, path)

        if not os.path.exists(full_path):
            return TaskResult(success=False, error_message="File not found")

        try:
            async with aiofiles.open(full_path, "r", encoding="utf-8") as f:
                content = await f.read()
            return TaskResult(success=True, result_data={"content": content})
        except Exception as e:
            return TaskResult(success=False, error_message=str(e))

    async def _list_dir(self, payload: Dict[str, Any]) -> TaskResult:
        path = payload.get("path", ".")

        if not self._is_safe_path(path):
            return TaskResult(
                success=False,
                error_message=f"Access denied: Path {path} is outside workspace",
            )

        full_path = os.path.join(self.root_dir, path)

        try:
            # Run blocking os.listdir in thread pool
            items = await asyncio.to_thread(os.listdir, full_path)
            return TaskResult(success=True, result_data={"items": items})
        except Exception as e:
            return TaskResult(success=False, error_message=str(e))

    async def _make_dir(self, payload: Dict[str, Any]) -> TaskResult:
        path = payload.get("path")

        if not path:
            return TaskResult(success=False, error_message="Missing path")

        if not self._is_safe_path(path):
            return TaskResult(
                success=False,
                error_message=f"Access denied: Path {path} is outside workspace",
            )

        full_path = os.path.join(self.root_dir, path)

        try:
            await asyncio.to_thread(os.makedirs, full_path, 0o755, True)
            return TaskResult(success=True, result_data={"path": path})
        except Exception as e:
            return TaskResult(success=False, error_message=str(e))
