"""
Audit Logger for Compliance
Structured JSON logging for forensics and compliance (SOC2, HIPAA, GDPR).
"""

import json
import logging
import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class AuditEvent(Enum):
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    WORKFLOW_CREATE = "workflow.create"
    WORKFLOW_COMPLETE = "workflow.complete"
    TASK_EXECUTE = "task.execute"
    FILE_READ = "file.read"
    FILE_WRITE = "file.write"
    CODE_EXECUTE = "code.execute"
    CHAT_MESSAGE = "chat.message"
    CONFIG_CHANGE = "config.change"
    SECURITY_VIOLATION = "security.violation"


class AuditLogger:
    """Structured audit logging to JSONL file."""

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self._logger = logging.getLogger("audit")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        if not self._logger.handlers:
            handler = logging.FileHandler(f"{log_dir}/audit.jsonl")
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    def log(
        self,
        event: AuditEvent,
        actor: str,
        resource: str,
        outcome: str = "success",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ):
        """Log an audit event."""
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": event.value,
            "actor": actor,
            "resource": resource,
            "outcome": outcome,
            "request_id": request_id,
            "ip": ip_address,
            "details": details or {},
        }
        self._logger.info(json.dumps(entry, default=str))


# Singleton
_audit: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    global _audit
    if _audit is None:
        _audit = AuditLogger()
    return _audit
