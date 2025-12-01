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
    api_key_enabled: bool = field(default_factory=lambda: os.getenv("ASTRO_API_KEY_ENABLED", "false").lower() == "true")
    api_keys: Set[str] = field(default_factory=lambda: set(filter(None, os.getenv("ASTRO_API_KEYS", "").split(","))))
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100  # requests per window
    rate_limit_window: int = 60     # seconds
    rate_limit_burst: int = 20      # burst allowance
    
    # CORS settings
    cors_origins: List[str] = field(default_factory=lambda: _parse_cors_origins())
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    cors_allow_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # Request limits
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: float = 30.0              # seconds
    
    # Exempt paths (no auth required)
    exempt_paths: Set[str] = field(default_factory=lambda: {
        "/", "/health", "/ready", "/metrics",
        "/styles.css", "/app.js", "/static",
        "/chat", "/workflows", "/vault", "/files",  # SPA routes
        "/docs", "/redoc", "/openapi.json",
    })
    
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
                (ts, count) for ts, count in self._requests[client_id]
                if ts > window_start
            ]
            
            # Count requests in current window
            current_count = sum(count for _, count in self._requests[client_id])
            
            # Calculate remaining
            remaining = max(0, self.requests_per_window - current_count)
            
            # Check burst (recent requests in last second)
            recent_start = now - 1.0
            recent_count = sum(
                count for ts, count in self._requests[client_id]
                if ts > recent_start
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
            logger.debug(f"Rate limiter cleanup: removed {len(to_remove)} inactive clients")


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
    """API key authentication middleware."""
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
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
            logger.warning(f"Missing API key for {request.method} {path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "API key required"},
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        if not verify_api_key(api_key):
            logger.warning(f"Invalid API key for {request.method} {path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid API key"},
                headers={"WWW-Authenticate": "ApiKey"},
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
    app.add_middleware(AuthenticationMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    
    logger.info(
        f"Security middleware configured: "
        f"auth={'enabled' if security_config.api_key_enabled else 'disabled'}, "
        f"rate_limit={'enabled' if security_config.rate_limit_enabled else 'disabled'}"
    )
