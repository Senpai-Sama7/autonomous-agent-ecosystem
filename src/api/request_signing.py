"""
HMAC Request Signing for API Security
Prevents replay attacks and ensures request integrity.
"""

import hmac
import hashlib
import os
import time
from typing import Tuple


class RequestSigner:
    """HMAC-SHA256 request signing."""

    TIMESTAMP_TOLERANCE = 300  # 5 minutes

    def __init__(self, secret_key: str = None):
        self.secret_key = (secret_key or os.getenv("ASTRO_SIGNING_KEY", "")).encode()

    def sign(self, method: str, path: str, body: str, timestamp: str) -> str:
        """Generate HMAC signature."""
        message = f"{timestamp}.{method}.{path}.{body}"
        return hmac.new(self.secret_key, message.encode(), hashlib.sha256).hexdigest()

    def verify(
        self, method: str, path: str, body: str, timestamp: str, signature: str
    ) -> Tuple[bool, str]:
        """Verify request signature. Returns (valid, error_message)."""
        if not self.secret_key:
            return True, "OK"  # Signing disabled if no key

        try:
            req_time = int(timestamp)
            if abs(time.time() - req_time) > self.TIMESTAMP_TOLERANCE:
                return False, "Request expired"
        except (ValueError, TypeError):
            return False, "Invalid timestamp"

        expected = self.sign(method, path, body, timestamp)
        if not hmac.compare_digest(expected, signature):
            return False, "Invalid signature"

        return True, "OK"


# Singleton
_signer = None


def get_request_signer() -> RequestSigner:
    global _signer
    if _signer is None:
        _signer = RequestSigner()
    return _signer
