"""
Helper utilities for data processing and validation
"""

import json
import re
from typing import Dict, Any, List, Optional
import hashlib


def sanitize_filename(filename: str) -> str:
    """Sanitize a string to be safe for use as a filename"""
    return re.sub(r"[^\w\-_\. ]", "_", filename)


def calculate_checksum(data: Any) -> str:
    """Calculate SHA-256 checksum of a dictionary or string"""
    if isinstance(data, dict):
        content = json.dumps(data, sort_keys=True)
    else:
        content = str(data)
    return hashlib.sha256(content.encode()).hexdigest()


def validate_json_schema(data: Dict, schema: Dict[str, type]) -> bool:
    """
    Validate that data conforms to a simple type schema.

    Args:
        data: Dictionary to validate
        schema: Dict mapping required keys to their expected Python types

    Returns:
        True if all required keys exist with correct types, False otherwise

    Example:
        >>> schema = {"name": str, "age": int}
        >>> validate_json_schema({"name": "Alice", "age": 30}, schema)
        True
        >>> validate_json_schema({"name": "Alice"}, schema)
        False

    Note:
        For complex JSON Schema validation (draft-07, etc.), consider using
        the 'jsonschema' library: pip install jsonschema
    """
    for key, expected_type in schema.items():
        if key not in data:
            return False
        if not isinstance(data[key], expected_type):
            return False
    return True


def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to max_length while preserving whole words if possible"""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."


def sanitize_display_text(text: str, max_length: int = 10000) -> str:
    """
    Sanitize text for safe display in GUI widgets.

    This function:
    - Removes control characters that could corrupt UI display
    - Preserves safe whitespace (newlines, tabs, spaces)
    - Truncates overly long text to prevent memory issues
    - Escapes potential problematic sequences

    Args:
        text: Input text to sanitize
        max_length: Maximum length of output text (default 10000)

    Returns:
        Sanitized text safe for GUI display

    Example:
        >>> sanitize_display_text("Hello\\x00World")
        'HelloWorld'
        >>> sanitize_display_text("Normal text\\nWith newlines")
        'Normal text\\nWith newlines'
    """
    if not text:
        return ""

    # Remove NULL bytes and other control characters (except \n, \r, \t)
    # Control chars are 0x00-0x1F and 0x7F, but we keep 0x09 (tab), 0x0A (newline), 0x0D (carriage return)
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Remove ANSI escape sequences (terminal color codes, etc.)
    sanitized = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", sanitized)

    # Remove other escape sequences
    sanitized = re.sub(r"\x1b[^a-zA-Z]*[a-zA-Z]", "", sanitized)

    # Limit length to prevent memory issues in GUI widgets
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "\n... (truncated)"

    return sanitized


def escape_log_message(message: str) -> str:
    """
    Escape a log message for safe inclusion in structured output.

    Args:
        message: Raw log message

    Returns:
        Escaped message safe for display
    """
    if not message:
        return ""

    # First sanitize control characters
    escaped = sanitize_display_text(message)

    # Escape any remaining special characters that might cause issues
    # in certain display contexts (but preserve readability)

    return escaped
