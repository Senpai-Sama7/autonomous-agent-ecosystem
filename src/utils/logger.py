"""
Centralized Logging Configuration for Autonomous Agent Ecosystem

This module provides a single source of truth for logging configuration.
All modules should use `get_logger(__name__)` instead of calling logging.basicConfig.

Usage:
    from utils.logger import configure_logging, get_logger
    
    # Call once at application startup (main.py or gui_app.py)
    configure_logging(log_level="INFO")
    
    # In any module
    logger = get_logger(__name__)
"""
import logging
import sys
import os
import threading
from logging.handlers import RotatingFileHandler
from typing import Optional

# Thread-safe initialization flag
_logging_configured = False
_logging_lock = threading.Lock()

# Default format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def configure_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_to_file: bool = True,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None
) -> None:
    """
    Configure the root logger for the entire application.
    
    This function should be called ONCE at application startup.
    Subsequent calls are no-ops to prevent duplicate handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (created if needed)
        log_to_file: Whether to also log to a rotating file
        log_format: Custom log format (uses DEFAULT_FORMAT if None)
        date_format: Custom date format (uses DEFAULT_DATE_FORMAT if None)
    """
    global _logging_configured
    
    with _logging_lock:
        if _logging_configured:
            return
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Clear any existing handlers (prevents duplicates)
        root_logger.handlers.clear()
        
        # Formatter
        formatter = logging.Formatter(
            log_format or DEFAULT_FORMAT,
            datefmt=date_format or DEFAULT_DATE_FORMAT
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        root_logger.addHandler(console_handler)
        
        # File Handler (Rotating) - optional
        if log_to_file:
            try:
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                    
                file_handler = RotatingFileHandler(
                    os.path.join(log_dir, "ecosystem.log"),
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5,
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
                root_logger.addHandler(file_handler)
            except (OSError, PermissionError) as e:
                # Log to console only if file logging fails
                root_logger.warning(f"Could not create log file: {e}. Logging to console only.")
        
        _logging_configured = True
        root_logger.debug("Logging configured successfully")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    This is the preferred way to get loggers throughout the codebase.
    If configure_logging() hasn't been called, a basic configuration is applied.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    # Auto-configure with defaults if not already configured
    if not _logging_configured:
        configure_logging()
    
    return logging.getLogger(name)


# Legacy function for backwards compatibility
def setup_logging(name: str, log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """
    DEPRECATED: Use configure_logging() at startup and get_logger() in modules.
    
    This function is kept for backwards compatibility but may add duplicate handlers
    if called multiple times with the same name.
    """
    import warnings
    warnings.warn(
        "setup_logging() is deprecated. Use configure_logging() at startup and get_logger() in modules.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Ensure global config is set
    configure_logging(log_level=log_level, log_dir=log_dir)
    
    return get_logger(name)
