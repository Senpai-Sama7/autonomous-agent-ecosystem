"""
ASTRO API Module
Provides REST and WebSocket API for the ASTRO web interface.

Exports:
- create_app: FastAPI application factory
- run_server: Convenience function to start the server
- add_security_middleware: Security middleware configuration
- security_config: Global security configuration
"""

from .server import create_app, run_server
from .middleware import (
    add_security_middleware,
    security_config,
    configure_security,
    get_allowed_origins,
)

__all__ = ["create_app", "run_server", "add_security_middleware", "security_config"]
