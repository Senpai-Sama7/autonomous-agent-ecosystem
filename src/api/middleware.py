"""
ASTRO API Middleware
Enterprise-grade middleware for security, rate limiting, and request handling.

This module provides:
- API Key authentication
- Rate limiting with sliding window
- Request ID tracking
- Security headers
- Request/response logging
"""

import asyncio
import hashlib
import hmac
import logging
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Callable, Dict, List, Optional, Set, Tuple

from fastapi import HTTPException, Request, Response, WebSocket, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger("ASTRO.Middleware")


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class SecurityConfig:
    """Security configuration with sensible defaults."""

    # API Key settings
    api_key_header: str = "X-API-Key"
    api_key_enabled: bool = field(
        default_factory=lambda: os.getenv("ASTRO_API_KEY_ENABLED", "false").lower()
        == "true"
    )
    api_keys: Set[str] = field(
        default_factory=lambda: set(
            filter(None, os.getenv("ASTRO_API_KEYS", "").split(","))
        )
    )

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100  # requests per window
    rate_limit_window: int = 60  # seconds
    rate_limit_burst: int = 20  # burst allowance

    # CORS settings
    cors_origins: List[str] = field(default_factory=lambda: _parse_cors_origins())
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    cors_allow_headers: List[str] = field(default_factory=lambda: ["*"])

    # Request limits
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: float = 30.0  # seconds

    # Exempt paths (no auth required)
    exempt_paths: Set[str] = field(
        default_factory=lambda: {
            "/",
            "/health",
            "/ready",
            "/metrics",
            "/styles.css",
            "/app.js",
            "/static",
            "/chat",
            "/workflows",
            "/vault",
            "/files",  # SPA routes
            "/docs",
            "/redoc",
            "/openapi.json",
        }
    )

    # WebSocket paths
    websocket_paths: Set[str] = field(default_factory=lambda: {"/ws"})


def _parse_cors_origins() -> List[str]:
    """Parse CORS origins from environment or use sensible defaults."""
    env_origins = os.getenv("ASTRO_CORS_ORIGINS", "")
    if env_origins:
        return [o.strip() for o in env_origins.split(",") if o.strip()]

    # Default: localhost variants for development
    return [
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:3000",
    ]


# Global config instance
security_config = SecurityConfig()


# ============================================================================
# RATE LIMITER
# ============================================================================


class SlidingWindowRateLimiter:
    """
    Token bucket rate limiter with sliding window.

    Features:
    - Per-client rate limiting by IP
    - Sliding window for smooth rate limiting
    - Burst allowance for legitimate traffic spikes
    - Automatic cleanup of old entries
    """

    def __init__(
        self,
        requests_per_window: int = 100,
        window_seconds: int = 60,
        burst_size: int = 20,
        cleanup_interval: int = 300,
    ):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.burst_size = burst_size
        self.cleanup_interval = cleanup_interval

        # Storage: client_id -> list of (timestamp, count) tuples
        self._requests: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        self._last_cleanup = time.time()
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed for client.

        Returns:
            Tuple of (allowed: bool, headers: dict with rate limit info)
        """
        async with self._lock:
            now = time.time()
            window_start = now - self.window_seconds

            # Clean old entries for this client
            self._requests[client_id] = [
                (ts, count)
                for ts, count in self._requests[client_id]
                if ts > window_start
            ]

            # Count requests in current window
            current_count = sum(count for _, count in self._requests[client_id])

            # Calculate remaining
            remaining = max(0, self.requests_per_window - current_count)

            # Check burst (recent requests in last second)
            recent_start = now - 1.0
            recent_count = sum(
                count for ts, count in self._requests[client_id] if ts > recent_start
            )

            headers = {
                "X-RateLimit-Limit": self.requests_per_window,
                "X-RateLimit-Remaining": remaining,
                "X-RateLimit-Reset": int(now + self.window_seconds),
            }

            # Check if allowed
            if current_count >= self.requests_per_window:
                headers["Retry-After"] = self.window_seconds
                return False, headers

            if recent_count >= self.burst_size:
                headers["Retry-After"] = 1
                return False, headers

            # Record this request
            self._requests[client_id].append((now, 1))

            # Periodic cleanup
            if now - self._last_cleanup > self.cleanup_interval:
                await self._cleanup(window_start)
                self._last_cleanup = now

            return True, headers

    async def _cleanup(self, cutoff: float):
        """Remove entries older than cutoff."""
        to_remove = []
        for client_id, requests in self._requests.items():
            self._requests[client_id] = [(ts, c) for ts, c in requests if ts > cutoff]
            if not self._requests[client_id]:
                to_remove.append(client_id)

        for client_id in to_remove:
            del self._requests[client_id]

        if to_remove:
            logger.debug(
                f"Rate limiter cleanup: removed {len(to_remove)} inactive clients"
            )


# Global rate limiter instance
rate_limiter = SlidingWindowRateLimiter(
    requests_per_window=security_config.rate_limit_requests,
    window_seconds=security_config.rate_limit_window,
    burst_size=security_config.rate_limit_burst,
)


# ============================================================================
# AUTHENTICATION
# ============================================================================


def verify_api_key(api_key: str) -> bool:
    """
    Verify API key using constant-time comparison.

    Uses hmac.compare_digest to prevent timing attacks.
    """
    if not security_config.api_keys:
        return False

    for valid_key in security_config.api_keys:
        if hmac.compare_digest(api_key.encode(), valid_key.encode()):
            return True
    return False


def get_client_id(request: Request) -> str:
    """
    Extract client identifier for rate limiting.

    Priority:
    1. X-Forwarded-For header (for proxied requests)
    2. Client host IP
    3. Fallback to 'unknown'
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


# ============================================================================
# MIDDLEWARE CLASSES
# ============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Don't cache API responses
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"

        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for tracing."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate or use existing request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Store in request state
        request.state.request_id = request_id

        # Call next middleware/handler
        response = await call_next(request)

        # Add to response headers
        response.headers["X-Request-ID"] = request_id

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using sliding window algorithm."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip rate limiting if disabled
        if not security_config.rate_limit_enabled:
            return await call_next(request)

        # Skip for exempt paths
        path = request.url.path
        if any(path.startswith(exempt) for exempt in security_config.exempt_paths):
            return await call_next(request)

        # Get client identifier
        client_id = get_client_id(request)

        # Check rate limit
        allowed, headers = await rate_limiter.is_allowed(client_id)

        if not allowed:
            logger.warning(f"Rate limit exceeded for client {client_id} on {path}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please slow down."},
                headers={k: str(v) for k, v in headers.items()},
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = str(value)

        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """API key authentication middleware with audit logging."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        from src.core.audit_logger import AuditEvent, get_audit_logger

        audit = get_audit_logger()
        client_ip = get_client_id(request)
        request_id = getattr(request.state, "request_id", None)

        # Skip auth if disabled
        if not security_config.api_key_enabled:
            return await call_next(request)

        # Skip for exempt paths
        path = request.url.path
        if any(path.startswith(exempt) for exempt in security_config.exempt_paths):
            return await call_next(request)

        # Skip for WebSocket upgrade requests (handled separately)
        if path in security_config.websocket_paths:
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get(security_config.api_key_header)

        if not api_key:
            audit.log(
                AuditEvent.AUTH_FAILURE,
                "anonymous",
                path,
                "failure",
                {"reason": "missing_key"},
                request_id,
                client_ip,
            )
            logger.warning(f"Missing API key for {request.method} {path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "API key required"},
                headers={"WWW-Authenticate": "ApiKey"},
            )

        if not verify_api_key(api_key):
            audit.log(
                AuditEvent.AUTH_FAILURE,
                "api_key",
                path,
                "failure",
                {"reason": "invalid_key"},
                request_id,
                client_ip,
            )
            logger.warning(f"Invalid API key for {request.method} {path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid API key"},
                headers={"WWW-Authenticate": "ApiKey"},
            )

        audit.log(
            AuditEvent.AUTH_SUCCESS,
            "api_key",
            path,
            "success",
            None,
            request_id,
            client_ip,
        )
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing information."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log request
        request_id = getattr(request.state, "request_id", "unknown")
        client_id = get_client_id(request)

        logger.info(
            f"{request.method} {request.url.path} - "
            f"{response.status_code} - "
            f"{duration_ms:.2f}ms - "
            f"client={client_id} - "
            f"request_id={request_id}"
        )

        return response


# ============================================================================
# WEBSOCKET AUTHENTICATION
# ============================================================================


async def authenticate_websocket(
    websocket: WebSocket,
    api_key: Optional[str] = None,
) -> bool:
    """
    Authenticate WebSocket connection.

    Checks API key from query parameter or first message.
    Returns True if authenticated or auth is disabled.
    """
    if not security_config.api_key_enabled:
        return True

    # Check query parameter first
    if api_key and verify_api_key(api_key):
        return True

    # Check header (some WebSocket clients support this)
    header_key = websocket.headers.get(security_config.api_key_header)
    if header_key and verify_api_key(header_key):
        return True

    return False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_allowed_origins() -> List[str]:
    """Get list of allowed CORS origins."""
    return security_config.cors_origins


def configure_security(
    api_key_enabled: Optional[bool] = None,
    rate_limit_enabled: Optional[bool] = None,
    cors_origins: Optional[List[str]] = None,
):
    """
    Configure security settings programmatically.

    This allows runtime configuration changes.
    """
    global security_config

    if api_key_enabled is not None:
        security_config.api_key_enabled = api_key_enabled

    if rate_limit_enabled is not None:
        security_config.rate_limit_enabled = rate_limit_enabled

    if cors_origins is not None:
        security_config.cors_origins = cors_origins


def add_security_middleware(app):
    """
    Add all security middleware to a FastAPI app.

    Order matters! Middleware is executed in reverse order of addition.
    """
    # Add in reverse order (last added = first executed)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(CSRFMiddleware)  # CSRF protection for state-changing requests
    app.add_middleware(AuthenticationMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

    logger.info(
        f"Security middleware configured: "
        f"auth={'enabled' if security_config.api_key_enabled else 'disabled'}, "
        f"csrf=enabled, "
        f"rate_limit={'enabled' if security_config.rate_limit_enabled else 'disabled'}"
    )


# ============================================================================
# CSRF PROTECTION
# ============================================================================


class CSRFProtection:
    """
    CSRF protection using double-submit cookie pattern.

    How it works:
    1. Server sets a CSRF token in a cookie (SameSite=Strict)
    2. Client must send the same token in X-CSRF-Token header
    3. Server validates both match

    This prevents CSRF because:
    - Attacker can't read the cookie (same-origin policy)
    - Attacker can't set custom headers on cross-origin requests
    """

    COOKIE_NAME = "csrf_token"
    HEADER_NAME = "X-CSRF-Token"
    TOKEN_LENGTH = 32

    def __init__(self):
        self._tokens: Dict[str, float] = {}  # token -> expiry timestamp
        self._token_ttl = 3600  # 1 hour

    def generate_token(self) -> str:
        """Generate a cryptographically secure CSRF token."""
        import secrets

        token = secrets.token_urlsafe(self.TOKEN_LENGTH)
        self._tokens[token] = time.time() + self._token_ttl
        self._cleanup_expired()
        return token

    def validate_token(
        self, cookie_token: Optional[str], header_token: Optional[str]
    ) -> bool:
        """Validate CSRF token using constant-time comparison."""
        if not cookie_token or not header_token:
            return False
        if cookie_token not in self._tokens:
            return False
        if self._tokens[cookie_token] < time.time():
            del self._tokens[cookie_token]
            return False
        return hmac.compare_digest(cookie_token, header_token)

    def _cleanup_expired(self):
        """Remove expired tokens."""
        now = time.time()
        self._tokens = {t: exp for t, exp in self._tokens.items() if exp > now}


csrf_protection = CSRFProtection()


class CSRFMiddleware(BaseHTTPMiddleware):
    """CSRF protection middleware for state-changing requests."""

    SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
    EXEMPT_PATHS = {
        "/health",
        "/ready",
        "/metrics",
        "/ws",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/",
    }

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip for safe methods
        if request.method in self.SAFE_METHODS:
            response = await call_next(request)
            # Set CSRF cookie on GET requests to pages
            if request.method == "GET" and not request.url.path.startswith("/api/"):
                token = csrf_protection.generate_token()
                response.set_cookie(
                    key=CSRFProtection.COOKIE_NAME,
                    value=token,
                    httponly=False,  # JS needs to read it
                    samesite="strict",
                    secure=os.getenv("ASTRO_ENV") == "production",
                    max_age=3600,
                )
            return response

        # Skip for exempt paths
        if any(request.url.path.startswith(p) for p in self.EXEMPT_PATHS):
            return await call_next(request)

        # Skip if API key auth is used (machine-to-machine)
        if request.headers.get(security_config.api_key_header):
            return await call_next(request)

        # Validate CSRF token
        cookie_token = request.cookies.get(CSRFProtection.COOKIE_NAME)
        header_token = request.headers.get(CSRFProtection.HEADER_NAME)

        if not csrf_protection.validate_token(cookie_token, header_token):
            logger.warning(
                f"CSRF validation failed for {request.method} {request.url.path}"
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "CSRF token missing or invalid"},
            )

        return await call_next(request)


# ============================================================================
# REQUEST SIGNING (HMAC)
# ============================================================================


class RequestSigningMiddleware(BaseHTTPMiddleware):
    """
    HMAC request signing middleware to prevent replay attacks.

    Required headers:
    - X-Timestamp: Unix timestamp
    - X-Signature: HMAC-SHA256 signature

    Enable by setting ASTRO_SIGNING_KEY environment variable.
    """

    EXEMPT_PATHS = {
        "/health",
        "/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/ws",
    }

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        from src.api.request_signing import get_request_signer

        signer = get_request_signer()

        # Skip if signing not configured
        if not signer.secret_key:
            return await call_next(request)

        # Skip exempt paths
        if any(request.url.path.startswith(p) for p in self.EXEMPT_PATHS):
            return await call_next(request)

        # Skip safe methods
        if request.method in {"GET", "HEAD", "OPTIONS"}:
            return await call_next(request)

        timestamp = request.headers.get("X-Timestamp", "")
        signature = request.headers.get("X-Signature", "")

        if not timestamp or not signature:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing signature headers"},
            )

        body = await request.body()
        valid, msg = signer.verify(
            request.method, request.url.path, body.decode(), timestamp, signature
        )

        if not valid:
            from src.core.audit_logger import AuditEvent, get_audit_logger

            audit = get_audit_logger()
            audit.log(
                AuditEvent.SECURITY_VIOLATION,
                "unknown",
                request.url.path,
                "failure",
                {"reason": msg},
                getattr(request.state, "request_id", None),
                get_client_id(request),
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": msg}
            )

        return await call_next(request)
