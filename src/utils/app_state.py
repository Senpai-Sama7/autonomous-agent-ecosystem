"""
App State Manager - Handles first-run detection, user preferences, and app state
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class AppState:
    """Manage application state and user preferences"""

    CONFIG_DIR = Path.home() / ".astro"
    CONFIG_FILE = CONFIG_DIR / "config.json"

    DEFAULT_STATE = {
        "first_run": True,
        "tutorial_completed": False,
        "theme": "dark",
        "last_provider": "openai",
        "last_model": "gpt-4o-mini",
        "window_maximized": False,
        "show_tips": True,
    }

    _instance = None
    _state: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """Load state from file"""
        try:
            self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            if self.CONFIG_FILE.exists():
                with open(self.CONFIG_FILE, "r") as f:
                    self._state = {**self.DEFAULT_STATE, **json.load(f)}
            else:
                self._state = self.DEFAULT_STATE.copy()
                self._save()
        except Exception:
            self._state = self.DEFAULT_STATE.copy()

    def _save(self):
        """Save state to file"""
        try:
            self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value"""
        return self._state.get(key, default)

    def set(self, key: str, value: Any):
        """Set a state value and save"""
        self._state[key] = value
        self._save()

    def is_first_run(self) -> bool:
        """Check if this is the first run"""
        return self._state.get("first_run", True)

    def mark_first_run_complete(self):
        """Mark first run as complete"""
        self._state["first_run"] = False
        self._state["tutorial_completed"] = True
        self._save()

    def reset_tutorial(self):
        """Reset tutorial to show again"""
        self._state["tutorial_completed"] = False
        self._save()
