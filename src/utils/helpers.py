"""
Helper utilities for data processing and validation
"""
import json
import re
from typing import Dict, Any, List, Optional
import hashlib

def sanitize_filename(filename: str) -> str:
    """Sanitize a string to be safe for use as a filename"""
    return re.sub(r'[^\w\-_\. ]', '_', filename)

def calculate_checksum(data: Any) -> str:
    """Calculate SHA-256 checksum of a dictionary or string"""
    if isinstance(data, dict):
        content = json.dumps(data, sort_keys=True)
    else:
        content = str(data)
    return hashlib.sha256(content.encode()).hexdigest()

def validate_json_schema(data: Dict, schema: Dict) -> bool:
    """Simple JSON schema validation (placeholder for jsonschema library)"""
    # In a real production system, use 'jsonschema' library
    for key, type_ in schema.items():
        if key not in data:
            return False
        if not isinstance(data[key], type_):
            return False
    return True

def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to max_length while preserving whole words if possible"""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + '...'
