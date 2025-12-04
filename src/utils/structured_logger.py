"""
Structured Logging System
JSON logs for ELK/Splunk/Datadog ingestion
"""
import asyncio
import functools
import json
import logging
import os
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Context variables for request tracing
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')
workflow_id_var: ContextVar[str] = ContextVar('workflow_id', default='')


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    SKIP_FIELDS = {
        'name', 'msg', 'args', 'created', 'filename', 'funcName', 
        'levelname', 'levelno', 'lineno', 'module', 'msecs', 'message',
        'pathname', 'process', 'processName', 'relativeCreated',
        'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "env": os.getenv("ENVIRONMENT", "development")
        }
        
        # Add context vars
        for name, var in [('req_id', request_id_var), ('user_id', user_id_var), ('wf_id', workflow_id_var)]:
            val = var.get()
            if val:
                log_data[name] = val
        
        # Add extra fields
        log_data.update({k: v for k, v in record.__dict__.items() if k not in self.SKIP_FIELDS})
        
        if record.exc_info:
            log_data['exc'] = {'type': record.exc_info[0].__name__, 'msg': str(record.exc_info[1])}
        
        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter for development."""
    
    COLORS = {'DEBUG': '\033[36m', 'INFO': '\033[32m', 'WARNING': '\033[33m', 
              'ERROR': '\033[31m', 'CRITICAL': '\033[35m', 'RESET': '\033[0m'}
    
    def format(self, record: logging.LogRecord) -> str:
        c = self.COLORS.get(record.levelname, '')
        r = self.COLORS['RESET']
        ts = datetime.now().strftime('%H:%M:%S')
        
        msg = f"{c}[{record.levelname}]{r} {ts} {record.name}:{record.lineno}"
        
        req_id = request_id_var.get()
        if req_id:
            msg += f" [{req_id[:8]}]"
        
        msg += f" {record.getMessage()}"
        
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        
        return msg


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None, 
                  json_format: bool = True, console: bool = True):
    """Setup structured logging."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper()))
    
    if console:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(StructuredFormatter() if json_format else ConsoleFormatter())
        root.addHandler(h)
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        h = logging.FileHandler(log_file)
        h.setFormatter(StructuredFormatter())
        root.addHandler(h)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class LogContext:
    """Context manager for setting request/workflow context."""
    
    def __init__(self, request_id: str = None, user_id: str = None, workflow_id: str = None):
        self.vals = [(request_id_var, request_id), (user_id_var, user_id), (workflow_id_var, workflow_id)]
        self.tokens = []
    
    def __enter__(self):
        for var, val in self.vals:
            if val:
                self.tokens.append((var, var.set(val)))
        return self
    
    def __exit__(self, *_):
        for var, token in self.tokens:
            try:
                var.reset(token)
            except (ValueError, LookupError):
                pass


def log_performance(func):
    """Decorator to log function execution time."""
    logger = get_logger(func.__module__)
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed", extra={'duration_ms': (time.time()-start)*1000, 'status': 'ok'})
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed", extra={'duration_ms': (time.time()-start)*1000, 'error': str(e)}, exc_info=True)
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed", extra={'duration_ms': (time.time()-start)*1000, 'status': 'ok'})
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed", extra={'duration_ms': (time.time()-start)*1000, 'error': str(e)}, exc_info=True)
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
