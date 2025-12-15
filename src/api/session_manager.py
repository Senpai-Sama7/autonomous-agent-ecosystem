"""
Session Manager with Token Rotation
Prevents session hijacking by rotating tokens periodically.
"""

import secrets
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: float
    last_rotated: float
    token: str


class SessionManager:
    """Session management with automatic token rotation."""

    ROTATION_INTERVAL = 900  # 15 minutes
    SESSION_TTL = 86400  # 24 hours

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create(self, user_id: str = "anonymous") -> Tuple[str, str]:
        """Create session. Returns (session_id, token)."""
        session_id = secrets.token_urlsafe(16)
        token = secrets.token_urlsafe(32)
        now = time.time()
        self._sessions[session_id] = Session(session_id, user_id, now, now, token)
        return session_id, token

    def validate(self, session_id: str, token: str) -> Tuple[bool, Optional[str]]:
        """Validate and rotate if needed. Returns (valid, new_token_or_none)."""
        session = self._sessions.get(session_id)
        if not session:
            return False, None

        if not secrets.compare_digest(session.token, token):
            return False, None

        if time.time() - session.created_at > self.SESSION_TTL:
            del self._sessions[session_id]
            return False, None

        # Rotate if needed
        new_token = None
        if time.time() - session.last_rotated > self.ROTATION_INTERVAL:
            new_token = secrets.token_urlsafe(32)
            session.token = new_token
            session.last_rotated = time.time()

        return True, new_token

    def invalidate(self, session_id: str):
        """Invalidate a session."""
        self._sessions.pop(session_id, None)

    def cleanup(self):
        """Remove expired sessions."""
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s.created_at > self.SESSION_TTL
        ]
        for sid in expired:
            del self._sessions[sid]


# Singleton
_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
