"""
ASTRO - Modern Chat Interface
A ChatGPT/Claude-like interface with agentic capabilities
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import asyncio
import queue
import logging
import sys
import os
import json
import re
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.engine import AgentEngine, Workflow, Task, WorkflowPriority
from core.nl_interface import NaturalLanguageInterface
from core.llm_factory import LLMFactory, UnifiedLLMClient
from monitoring.monitoring_dashboard import MonitoringDashboard
from main import AutonomousAgentEcosystem
from utils.model_manager import ModelManager, ModelInfo
from utils.app_state import AppState
from utils.helpers import sanitize_display_text

# Configure CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# Theme Configuration
class Theme:
    """Modern dark theme inspired by Claude.ai and ChatGPT"""

    # Backgrounds
    BG_PRIMARY = "#0d0d0d"  # Main background
    BG_SECONDARY = "#171717"  # Sidebar background
    BG_TERTIARY = "#1a1a1a"  # Card background
    BG_INPUT = "#262626"  # Input field background
    BG_HOVER = "#2a2a2a"  # Hover state
    BG_MESSAGE_USER = "#2563eb"  # User message bubble
    BG_MESSAGE_ASSISTANT = "#262626"  # Assistant message bubble

    # Borders
    BORDER = "#333333"
    BORDER_LIGHT = "#404040"
    BORDER_FOCUS = "#3b82f6"

    # Accent colors
    ACCENT = "#3b82f6"  # Blue 500
    ACCENT_HOVER = "#2563eb"  # Blue 600
    ACCENT_LIGHT = "#60a5fa"  # Blue 400

    # Status colors
    SUCCESS = "#22c55e"
    WARNING = "#f59e0b"
    ERROR = "#ef4444"
    INFO = "#3b82f6"

    # Text colors
    TEXT_PRIMARY = "#fafafa"
    TEXT_SECONDARY = "#a3a3a3"
    TEXT_MUTED = "#737373"
    TEXT_INVERSE = "#0d0d0d"

    # Code colors
    CODE_BG = "#1e1e1e"
    CODE_BORDER = "#333333"

    # Font sizes
    FONT_XS = 11
    FONT_SM = 12
    FONT_BASE = 13
    FONT_LG = 14
    FONT_XL = 16
    FONT_2XL = 20
    FONT_3XL = 24


@dataclass
class Message:
    """Chat message"""

    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    model: str = ""
    is_streaming: bool = False
    agent_actions: List[Dict] = field(default_factory=list)


@dataclass
class Conversation:
    """Conversation thread"""

    id: str
    title: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    model: str = ""
    provider: str = ""


class ConversationStore:
    """Manages conversation history"""

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(__file__), "..", "data", "conversations.json"
        )
        self.conversations: Dict[str, Conversation] = {}
        self.current_id: Optional[str] = None
        self._load()

    def _load(self):
        """Load conversations from disk"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    for conv_data in data.get("conversations", []):
                        conv = Conversation(
                            id=conv_data["id"],
                            title=conv_data["title"],
                            created_at=datetime.fromisoformat(conv_data["created_at"]),
                            updated_at=datetime.fromisoformat(conv_data["updated_at"]),
                            model=conv_data.get("model", ""),
                            provider=conv_data.get("provider", ""),
                        )
                        for msg_data in conv_data.get("messages", []):
                            msg = Message(
                                id=msg_data["id"],
                                role=msg_data["role"],
                                content=msg_data["content"],
                                timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                                model=msg_data.get("model", ""),
                            )
                            conv.messages.append(msg)
                        self.conversations[conv.id] = conv
                    self.current_id = data.get("current_id")
        except Exception as e:
            logging.error(f"Error loading conversations: {e}")

    def _save(self):
        """Save conversations to disk"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            data = {
                "current_id": self.current_id,
                "conversations": [
                    {
                        "id": conv.id,
                        "title": conv.title,
                        "created_at": conv.created_at.isoformat(),
                        "updated_at": conv.updated_at.isoformat(),
                        "model": conv.model,
                        "provider": conv.provider,
                        "messages": [
                            {
                                "id": msg.id,
                                "role": msg.role,
                                "content": msg.content,
                                "timestamp": msg.timestamp.isoformat(),
                                "model": msg.model,
                            }
                            for msg in conv.messages
                        ],
                    }
                    for conv in self.conversations.values()
                ],
            }
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving conversations: {e}")

    def create(self, title: str = "New Chat") -> Conversation:
        """Create a new conversation"""
        conv = Conversation(id=str(uuid.uuid4()), title=title)
        self.conversations[conv.id] = conv
        self.current_id = conv.id
        self._save()
        return conv

    def get_current(self) -> Optional[Conversation]:
        """Get current conversation"""
        if self.current_id and self.current_id in self.conversations:
            return self.conversations[self.current_id]
        return None

    def set_current(self, conv_id: str):
        """Set current conversation"""
        if conv_id in self.conversations:
            self.current_id = conv_id
            self._save()

    def add_message(self, conv_id: str, message: Message):
        """Add message to conversation"""
        if conv_id in self.conversations:
            conv = self.conversations[conv_id]
            conv.messages.append(message)
            conv.updated_at = datetime.now()
            # Update title from first user message
            if len(conv.messages) == 1 and message.role == "user":
                conv.title = message.content[:50] + ("..." if len(message.content) > 50 else "")
            self._save()

    def update_message(self, conv_id: str, msg_id: str, content: str):
        """Update message content"""
        if conv_id in self.conversations:
            for msg in self.conversations[conv_id].messages:
                if msg.id == msg_id:
                    msg.content = content
                    self._save()
                    break

    def delete(self, conv_id: str):
        """Delete a conversation"""
        if conv_id in self.conversations:
            del self.conversations[conv_id]
            if self.current_id == conv_id:
                self.current_id = None
            self._save()

    def get_all(self) -> List[Conversation]:
        """Get all conversations sorted by updated_at"""
        return sorted(self.conversations.values(), key=lambda c: c.updated_at, reverse=True)


class MessageBubble(ctk.CTkFrame):
    """Chat message bubble widget"""

    def __init__(self, master, message: Message, on_copy=None, on_regenerate=None, **kwargs):
        super().__init__(master, **kwargs)
        self.message = message
        self.on_copy = on_copy
        self.on_regenerate = on_regenerate

        is_user = message.role == "user"

        self.configure(
            fg_color=Theme.BG_MESSAGE_USER if is_user else Theme.BG_MESSAGE_ASSISTANT,
            corner_radius=16,
        )

        # Content container
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(fill="x", padx=16, pady=12)

        # Role indicator (for assistant)
        if not is_user:
            role_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
            role_frame.pack(fill="x", pady=(0, 8))

            ctk.CTkLabel(
                role_frame,
                text="ASTRO",
                font=ctk.CTkFont(size=Theme.FONT_SM, weight="bold"),
                text_color=Theme.ACCENT_LIGHT,
            ).pack(side="left")

            if message.model:
                ctk.CTkLabel(
                    role_frame,
                    text=f" · {message.model}",
                    font=ctk.CTkFont(size=Theme.FONT_XS),
                    text_color=Theme.TEXT_MUTED,
                ).pack(side="left")

        # Message content with code block handling
        self._render_content(content_frame, message.content, is_user)

        # Action buttons (for assistant messages)
        if not is_user and not message.is_streaming:
            actions_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
            actions_frame.pack(fill="x", pady=(12, 0))

            # Copy button
            copy_btn = ctk.CTkButton(
                actions_frame,
                text="Copy",
                width=60,
                height=28,
                corner_radius=6,
                fg_color="transparent",
                hover_color=Theme.BG_HOVER,
                text_color=Theme.TEXT_MUTED,
                font=ctk.CTkFont(size=Theme.FONT_XS),
                command=lambda: self._copy_content(),
            )
            copy_btn.pack(side="left", padx=(0, 4))

            # Regenerate button
            regen_btn = ctk.CTkButton(
                actions_frame,
                text="Regenerate",
                width=80,
                height=28,
                corner_radius=6,
                fg_color="transparent",
                hover_color=Theme.BG_HOVER,
                text_color=Theme.TEXT_MUTED,
                font=ctk.CTkFont(size=Theme.FONT_XS),
                command=lambda: self._regenerate(),
            )
            regen_btn.pack(side="left")

    def _render_content(self, parent, content: str, is_user: bool):
        """Render message content with code block support"""
        # Simple code block detection
        code_pattern = r"```(\w*)\n?([\s\S]*?)```"

        last_end = 0
        for match in re.finditer(code_pattern, content):
            # Text before code block
            if match.start() > last_end:
                text = content[last_end : match.start()].strip()
                if text:
                    self._add_text(parent, text, is_user)

            # Code block
            lang = match.group(1) or "text"
            code = match.group(2).strip()
            self._add_code_block(parent, code, lang)

            last_end = match.end()

        # Remaining text after last code block
        if last_end < len(content):
            text = content[last_end:].strip()
            if text:
                self._add_text(parent, text, is_user)

    def _add_text(self, parent, text: str, is_user: bool):
        """Add text content"""
        label = ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=Theme.FONT_BASE),
            text_color=Theme.TEXT_PRIMARY if is_user else Theme.TEXT_PRIMARY,
            wraplength=600,
            justify="left",
            anchor="w",
        )
        label.pack(fill="x", anchor="w")

    def _add_code_block(self, parent, code: str, lang: str):
        """Add code block with copy button"""
        code_frame = ctk.CTkFrame(
            parent,
            fg_color=Theme.CODE_BG,
            corner_radius=8,
            border_width=1,
            border_color=Theme.CODE_BORDER,
        )
        code_frame.pack(fill="x", pady=8)

        # Header with language and copy button
        header = ctk.CTkFrame(code_frame, fg_color="#2d2d2d", corner_radius=0)
        header.pack(fill="x")

        ctk.CTkLabel(
            header,
            text=lang.upper() if lang else "CODE",
            font=ctk.CTkFont(size=Theme.FONT_XS),
            text_color=Theme.TEXT_MUTED,
        ).pack(side="left", padx=12, pady=6)

        copy_code_btn = ctk.CTkButton(
            header,
            text="Copy",
            width=50,
            height=24,
            corner_radius=4,
            fg_color="transparent",
            hover_color=Theme.BG_HOVER,
            text_color=Theme.TEXT_MUTED,
            font=ctk.CTkFont(size=Theme.FONT_XS),
            command=lambda c=code: self._copy_to_clipboard(c),
        )
        copy_code_btn.pack(side="right", padx=8, pady=4)

        # Code content
        code_text = ctk.CTkTextbox(
            code_frame,
            font=("SF Mono", Theme.FONT_SM),
            fg_color=Theme.CODE_BG,
            text_color="#e0e0e0",
            wrap="none",
            height=min(200, 20 + code.count("\n") * 18),
        )
        code_text.pack(fill="x", padx=12, pady=(0, 12))
        code_text.insert("1.0", code)
        code_text.configure(state="disabled")

    def _copy_content(self):
        """Copy full message content"""
        self._copy_to_clipboard(self.message.content)
        if self.on_copy:
            self.on_copy()

    def _copy_to_clipboard(self, text: str):
        """Copy text to clipboard"""
        self.clipboard_clear()
        self.clipboard_append(text)

    def _regenerate(self):
        """Regenerate response"""
        if self.on_regenerate:
            self.on_regenerate(self.message)


class ConversationListItem(ctk.CTkFrame):
    """Sidebar conversation item"""

    def __init__(self, master, conversation: Conversation, on_select=None, on_delete=None, is_active=False, **kwargs):
        super().__init__(master, **kwargs)
        self.conversation = conversation
        self.on_select = on_select
        self.on_delete = on_delete

        self.configure(
            fg_color=Theme.BG_HOVER if is_active else "transparent",
            corner_radius=8,
            cursor="hand2",
        )

        # Bind click
        self.bind("<Button-1>", self._on_click)

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="x", padx=12, pady=10)
        content.bind("<Button-1>", self._on_click)

        # Title
        title_label = ctk.CTkLabel(
            content,
            text=conversation.title[:30] + ("..." if len(conversation.title) > 30 else ""),
            font=ctk.CTkFont(size=Theme.FONT_SM),
            text_color=Theme.TEXT_PRIMARY,
            anchor="w",
        )
        title_label.pack(fill="x")
        title_label.bind("<Button-1>", self._on_click)

        # Timestamp
        time_str = conversation.updated_at.strftime("%b %d")
        ctk.CTkLabel(
            content,
            text=time_str,
            font=ctk.CTkFont(size=Theme.FONT_XS),
            text_color=Theme.TEXT_MUTED,
            anchor="w",
        ).pack(fill="x")

        # Delete button (shown on hover)
        self.delete_btn = ctk.CTkButton(
            self,
            text="x",
            width=24,
            height=24,
            corner_radius=4,
            fg_color="transparent",
            hover_color=Theme.ERROR,
            text_color=Theme.TEXT_MUTED,
            font=ctk.CTkFont(size=Theme.FONT_SM),
            command=self._on_delete,
        )
        self.delete_btn.place(relx=1.0, rely=0.5, anchor="e", x=-8)
        self.delete_btn.place_forget()

        # Show delete on hover
        self.bind("<Enter>", self._show_delete)
        self.bind("<Leave>", self._hide_delete)

    def _on_click(self, event=None):
        if self.on_select:
            self.on_select(self.conversation.id)

    def _on_delete(self):
        if self.on_delete:
            self.on_delete(self.conversation.id)

    def _show_delete(self, event=None):
        self.delete_btn.place(relx=1.0, rely=0.5, anchor="e", x=-8)

    def _hide_delete(self, event=None):
        self.delete_btn.place_forget()


class SettingsPanel(ctk.CTkToplevel):
    """Settings panel with provider and model selection"""

    def __init__(self, master, on_save=None):
        super().__init__(master)
        self.on_save = on_save

        self.title("Settings")
        self.geometry("550x700")
        self.configure(fg_color=Theme.BG_PRIMARY)
        self.transient(master)

        self.models_cache = {}
        self._build_ui()

        self.after(100, self._setup_grab)

    def _setup_grab(self):
        try:
            if self.winfo_exists():
                self.grab_set()
                self.focus_force()
        except Exception:
            pass

    def _build_ui(self):
        """Build settings UI"""
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=24, pady=24)

        # Header
        ctk.CTkLabel(
            scroll,
            text="Settings",
            font=ctk.CTkFont(size=Theme.FONT_3XL, weight="bold"),
            text_color=Theme.TEXT_PRIMARY,
        ).pack(anchor="w", pady=(0, 4))

        ctk.CTkLabel(
            scroll,
            text="Configure your AI provider, model, and preferences",
            font=ctk.CTkFont(size=Theme.FONT_BASE),
            text_color=Theme.TEXT_MUTED,
        ).pack(anchor="w", pady=(0, 24))

        # Provider Selection
        self._create_section_header(scroll, "AI Provider")

        provider_frame = ctk.CTkFrame(scroll, fg_color=Theme.BG_TERTIARY, corner_radius=12)
        provider_frame.pack(fill="x", pady=(0, 20))

        self.provider_var = ctk.StringVar(value=os.getenv("LLM_PROVIDER", "openai"))
        self.provider_var.trace_add("write", self._on_provider_change)

        providers = ModelManager.get_all_providers()
        for i, provider in enumerate(providers):
            self._create_provider_option(
                provider_frame,
                provider["id"],
                provider["name"],
                provider["description"],
                provider.get("free_tier", False),
                i == 0,
            )

        # Model Selection
        self._create_section_header(scroll, "Model")

        model_frame = ctk.CTkFrame(scroll, fg_color=Theme.BG_TERTIARY, corner_radius=12)
        model_frame.pack(fill="x", pady=(0, 20))

        model_content = ctk.CTkFrame(model_frame, fg_color="transparent")
        model_content.pack(fill="x", padx=16, pady=16)

        self.model_var = ctk.StringVar(value=os.getenv("LLM_MODEL", "gpt-4o-mini"))
        self.model_dropdown = ctk.CTkOptionMenu(
            model_content,
            variable=self.model_var,
            values=["Loading..."],
            fg_color=Theme.BG_INPUT,
            button_color=Theme.ACCENT,
            button_hover_color=Theme.ACCENT_HOVER,
            dropdown_fg_color=Theme.BG_TERTIARY,
            dropdown_hover_color=Theme.BG_HOVER,
            text_color=Theme.TEXT_PRIMARY,
            height=44,
            corner_radius=8,
            font=ctk.CTkFont(size=Theme.FONT_BASE),
            command=self._on_model_change,
        )
        self.model_dropdown.pack(fill="x")

        self.model_info_label = ctk.CTkLabel(
            model_content,
            text="",
            font=ctk.CTkFont(size=Theme.FONT_XS),
            text_color=Theme.TEXT_MUTED,
        )
        self.model_info_label.pack(anchor="w", pady=(8, 0))

        # API Key Section
        self.api_section = ctk.CTkFrame(scroll, fg_color="transparent")
        self.api_section.pack(fill="x")

        self._create_section_header(self.api_section, "API Key")

        api_frame = ctk.CTkFrame(self.api_section, fg_color=Theme.BG_TERTIARY, corner_radius=12)
        api_frame.pack(fill="x", pady=(0, 20))

        api_content = ctk.CTkFrame(api_frame, fg_color="transparent")
        api_content.pack(fill="x", padx=16, pady=16)

        self.api_key_entry = ctk.CTkEntry(
            api_content,
            show="*",
            placeholder_text="Enter your API key",
            fg_color=Theme.BG_INPUT,
            border_color=Theme.BORDER,
            text_color=Theme.TEXT_PRIMARY,
            height=44,
            corner_radius=8,
        )
        self.api_key_entry.pack(fill="x")

        # Load existing key
        provider = self.provider_var.get()
        key_env = self._get_key_env(provider)
        if key_env:
            existing_key = os.getenv(key_env, "")
            if existing_key:
                self.api_key_entry.insert(0, existing_key)

        ctk.CTkLabel(
            api_content,
            text="Your API key is stored locally and never shared",
            font=ctk.CTkFont(size=Theme.FONT_XS),
            text_color=Theme.TEXT_MUTED,
        ).pack(anchor="w", pady=(8, 0))

        # Status indicator
        self.status_frame = ctk.CTkFrame(scroll, fg_color=Theme.BG_TERTIARY, corner_radius=12)
        self.status_frame.pack(fill="x", pady=(0, 20))

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=ctk.CTkFont(size=Theme.FONT_SM),
            text_color=Theme.TEXT_SECONDARY,
        )
        self.status_label.pack(padx=16, pady=12)

        # Save Button
        ctk.CTkButton(
            scroll,
            text="Save Settings",
            command=self._save,
            fg_color=Theme.ACCENT,
            hover_color=Theme.ACCENT_HOVER,
            height=50,
            corner_radius=10,
            font=ctk.CTkFont(size=Theme.FONT_LG, weight="bold"),
        ).pack(fill="x", pady=(10, 0))

        # Load initial models
        self._load_models()

    def _create_section_header(self, parent, title: str):
        """Create section header"""
        ctk.CTkLabel(
            parent,
            text=title,
            font=ctk.CTkFont(size=Theme.FONT_LG, weight="bold"),
            text_color=Theme.TEXT_PRIMARY,
        ).pack(anchor="w", pady=(0, 8))

    def _create_provider_option(self, parent, value: str, name: str, desc: str, free_tier: bool, is_first: bool):
        """Create provider radio option"""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=16, pady=(16 if is_first else 0, 0 if is_first else 16))

        radio = ctk.CTkRadioButton(
            frame,
            text="",
            variable=self.provider_var,
            value=value,
            fg_color=Theme.ACCENT,
            border_color=Theme.BORDER,
            hover_color=Theme.ACCENT_HOVER,
        )
        radio.pack(side="left")

        text_frame = ctk.CTkFrame(frame, fg_color="transparent")
        text_frame.pack(side="left", padx=(8, 0), fill="x", expand=True)

        name_frame = ctk.CTkFrame(text_frame, fg_color="transparent")
        name_frame.pack(anchor="w")

        ctk.CTkLabel(
            name_frame,
            text=name,
            font=ctk.CTkFont(size=Theme.FONT_BASE, weight="bold"),
            text_color=Theme.TEXT_PRIMARY,
        ).pack(side="left")

        if free_tier:
            ctk.CTkLabel(
                name_frame,
                text=" FREE",
                font=ctk.CTkFont(size=Theme.FONT_XS, weight="bold"),
                text_color=Theme.SUCCESS,
            ).pack(side="left", padx=(8, 0))

        ctk.CTkLabel(
            text_frame,
            text=desc,
            font=ctk.CTkFont(size=Theme.FONT_XS),
            text_color=Theme.TEXT_MUTED,
        ).pack(anchor="w")

    def _get_key_env(self, provider: str) -> str:
        """Get environment variable name for provider API key"""
        mapping = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        return mapping.get(provider, "")

    def _on_provider_change(self, *args):
        """Handle provider change"""
        provider = self.provider_var.get()

        # Show/hide API key section
        if provider == "ollama":
            self.api_section.pack_forget()
        else:
            self.api_section.pack(fill="x", before=self.status_frame)
            # Update key entry with appropriate key
            key_env = self._get_key_env(provider)
            self.api_key_entry.delete(0, "end")
            if key_env:
                existing_key = os.getenv(key_env, "")
                if existing_key:
                    self.api_key_entry.insert(0, existing_key)

        self._load_models()
        self._check_status()

    def _on_model_change(self, value):
        """Handle model change"""
        self._update_model_info()

    def _load_models(self):
        """Load models for current provider"""
        provider = self.provider_var.get()
        models = ModelManager.get_models_for_provider(provider)

        if models:
            self.models_cache = {m.name: m for m in models}
            model_names = [m.name for m in models]
            self.model_dropdown.configure(values=model_names)

            # Set current or default model
            current = os.getenv("LLM_MODEL", "")
            model_ids = [m.id for m in models]

            if current in model_ids:
                idx = model_ids.index(current)
                self.model_var.set(model_names[idx])
            else:
                self.model_var.set(model_names[0])

            self._update_model_info()
        else:
            self.model_dropdown.configure(values=["No models available"])
            self.model_var.set("No models available")

    def _update_model_info(self):
        """Update model info display"""
        model_name = self.model_var.get()
        if model_name in self.models_cache:
            model = self.models_cache[model_name]
            info_parts = [model.description]
            if model.context_window:
                info_parts.append(f"{model.context_window:,} tokens")
            if model.supports_vision:
                info_parts.append("Vision")
            if model.supports_tools:
                info_parts.append("Tools")
            self.model_info_label.configure(text=" · ".join(info_parts))

    def _check_status(self):
        """Check provider status"""
        provider = self.provider_var.get()

        if provider == "ollama":
            status = ModelManager.check_ollama_status()
            if status["running"]:
                self.status_label.configure(
                    text=f"Ollama running with {status['model_count']} models",
                    text_color=Theme.SUCCESS,
                )
            else:
                self.status_label.configure(
                    text="Ollama not running. Start with: ollama serve",
                    text_color=Theme.WARNING,
                )
        elif provider == "gemini":
            api_key = self.api_key_entry.get() or os.getenv("GEMINI_API_KEY")
            if api_key:
                status = ModelManager.check_gemini_status(api_key)
                if status.get("available"):
                    self.status_label.configure(
                        text=f"Gemini API connected ({status['model_count']} models)",
                        text_color=Theme.SUCCESS,
                    )
                else:
                    self.status_label.configure(
                        text=f"Error: {status.get('error', 'Unknown')}",
                        text_color=Theme.ERROR,
                    )
            else:
                self.status_label.configure(
                    text="Enter your Gemini API key",
                    text_color=Theme.TEXT_MUTED,
                )
        else:
            self.status_label.configure(text="", text_color=Theme.TEXT_MUTED)

    def _save(self):
        """Save settings"""
        provider = self.provider_var.get()
        os.environ["LLM_PROVIDER"] = provider

        # Get model ID
        model_name = self.model_var.get()
        if model_name in self.models_cache:
            os.environ["LLM_MODEL"] = self.models_cache[model_name].id
        else:
            os.environ["LLM_MODEL"] = model_name

        # Save API key
        if provider != "ollama":
            api_key = self.api_key_entry.get()
            if api_key:
                key_env = self._get_key_env(provider)
                if key_env:
                    os.environ[key_env] = api_key

        # Provider-specific settings
        if provider == "ollama":
            os.environ["LLM_BASE_URL"] = "http://localhost:11434/v1"

        if self.on_save:
            self.on_save()

        logging.info(f"Settings saved: {provider} / {os.getenv('LLM_MODEL')}")
        self.destroy()


class ModernChatApp(ctk.CTk):
    """Modern chat interface for ASTRO"""

    def __init__(self):
        super().__init__()

        self.title("ASTRO")
        self.geometry("1400x900")
        self.minsize(1000, 700)
        self.configure(fg_color=Theme.BG_PRIMARY)

        # State
        self.conversation_store = ConversationStore()
        self.message_widgets: Dict[str, MessageBubble] = {}
        self.is_streaming = False
        self.current_stream_msg_id: Optional[str] = None

        # Engine state
        self.ecosystem = None
        self.llm_client = None
        self.nl_interface = None
        self.loop = None
        self.running = False
        self.log_queue = queue.Queue()

        # Build UI
        self._build_ui()

        # Setup
        self._setup_logging()
        self._setup_keybindings()

        # Start background tasks
        self.after(100, self._process_logs)
        self.after(500, self._initialize_engine)

    def _build_ui(self):
        """Build the main UI"""
        # Main container
        self.grid_columnconfigure(0, weight=0)  # Sidebar
        self.grid_columnconfigure(1, weight=1)  # Main area
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self._build_sidebar()

        # Main chat area
        self._build_main_area()

    def _build_sidebar(self):
        """Build sidebar with conversation list"""
        sidebar = ctk.CTkFrame(
            self,
            width=280,
            fg_color=Theme.BG_SECONDARY,
            corner_radius=0,
        )
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        # Header
        header = ctk.CTkFrame(sidebar, fg_color="transparent", height=60)
        header.pack(fill="x", padx=16, pady=16)
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="ASTRO",
            font=ctk.CTkFont(size=Theme.FONT_2XL, weight="bold"),
            text_color=Theme.TEXT_PRIMARY,
        ).pack(side="left")

        # New chat button
        new_chat_btn = ctk.CTkButton(
            header,
            text="+",
            width=36,
            height=36,
            corner_radius=8,
            fg_color=Theme.ACCENT,
            hover_color=Theme.ACCENT_HOVER,
            font=ctk.CTkFont(size=Theme.FONT_XL, weight="bold"),
            command=self._new_conversation,
        )
        new_chat_btn.pack(side="right")

        # Conversation list
        self.conv_list_frame = ctk.CTkScrollableFrame(
            sidebar,
            fg_color="transparent",
            scrollbar_button_color=Theme.BORDER,
        )
        self.conv_list_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Bottom section
        bottom = ctk.CTkFrame(sidebar, fg_color="transparent")
        bottom.pack(fill="x", padx=16, pady=16)

        # Settings button
        settings_btn = ctk.CTkButton(
            bottom,
            text="Settings",
            fg_color=Theme.BG_TERTIARY,
            hover_color=Theme.BG_HOVER,
            text_color=Theme.TEXT_SECONDARY,
            height=40,
            corner_radius=8,
            command=self._open_settings,
        )
        settings_btn.pack(fill="x")

        # Provider/model display
        self.provider_label = ctk.CTkLabel(
            bottom,
            text="",
            font=ctk.CTkFont(size=Theme.FONT_XS),
            text_color=Theme.TEXT_MUTED,
        )
        self.provider_label.pack(pady=(8, 0))

        self._refresh_conversation_list()
        self._update_provider_display()

    def _build_main_area(self):
        """Build main chat area"""
        main = ctk.CTkFrame(self, fg_color=Theme.BG_PRIMARY, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=0)  # Header
        main.grid_rowconfigure(1, weight=1)  # Messages
        main.grid_rowconfigure(2, weight=0)  # Input
        main.grid_columnconfigure(0, weight=1)

        # Header with model selector
        header = ctk.CTkFrame(main, fg_color=Theme.BG_SECONDARY, height=56, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)

        header_content = ctk.CTkFrame(header, fg_color="transparent")
        header_content.pack(fill="x", padx=20, pady=12)

        # Model selector dropdown
        self.model_selector_var = ctk.StringVar(value="Select Model")
        self.model_selector = ctk.CTkOptionMenu(
            header_content,
            variable=self.model_selector_var,
            values=["Loading..."],
            fg_color=Theme.BG_TERTIARY,
            button_color=Theme.BG_HOVER,
            button_hover_color=Theme.BORDER,
            dropdown_fg_color=Theme.BG_TERTIARY,
            dropdown_hover_color=Theme.BG_HOVER,
            text_color=Theme.TEXT_PRIMARY,
            width=250,
            height=32,
            corner_radius=6,
            font=ctk.CTkFont(size=Theme.FONT_SM),
            command=self._on_model_select,
        )
        self.model_selector.pack(side="left")

        # Status indicator
        self.status_dot = ctk.CTkLabel(
            header_content,
            text="●",
            font=ctk.CTkFont(size=10),
            text_color=Theme.TEXT_MUTED,
        )
        self.status_dot.pack(side="right")

        self.status_text = ctk.CTkLabel(
            header_content,
            text="Ready",
            font=ctk.CTkFont(size=Theme.FONT_SM),
            text_color=Theme.TEXT_MUTED,
        )
        self.status_text.pack(side="right", padx=(0, 8))

        # Messages area
        self.messages_frame = ctk.CTkScrollableFrame(
            main,
            fg_color="transparent",
            scrollbar_button_color=Theme.BORDER,
        )
        self.messages_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)

        # Welcome message (shown when no messages)
        self.welcome_frame = ctk.CTkFrame(self.messages_frame, fg_color="transparent")
        self.welcome_frame.pack(expand=True, pady=100)

        ctk.CTkLabel(
            self.welcome_frame,
            text="Welcome to ASTRO",
            font=ctk.CTkFont(size=Theme.FONT_3XL, weight="bold"),
            text_color=Theme.TEXT_PRIMARY,
        ).pack()

        ctk.CTkLabel(
            self.welcome_frame,
            text="Your AI assistant with agentic capabilities",
            font=ctk.CTkFont(size=Theme.FONT_LG),
            text_color=Theme.TEXT_MUTED,
        ).pack(pady=(8, 24))

        # Quick actions
        quick_frame = ctk.CTkFrame(self.welcome_frame, fg_color="transparent")
        quick_frame.pack()

        quick_actions = [
            ("Research a topic", "Research the latest advances in AI"),
            ("Write code", "Write a Python function to sort a list"),
            ("Create a file", "Create a todo list and save it"),
        ]

        for title, prompt in quick_actions:
            btn = ctk.CTkButton(
                quick_frame,
                text=title,
                fg_color=Theme.BG_TERTIARY,
                hover_color=Theme.BG_HOVER,
                text_color=Theme.TEXT_SECONDARY,
                height=40,
                corner_radius=8,
                font=ctk.CTkFont(size=Theme.FONT_SM),
                command=lambda p=prompt: self._set_input(p),
            )
            btn.pack(side="left", padx=4)

        # Input area
        input_container = ctk.CTkFrame(main, fg_color="transparent")
        input_container.grid(row=2, column=0, sticky="ew", padx=20, pady=20)

        input_frame = ctk.CTkFrame(
            input_container,
            fg_color=Theme.BG_INPUT,
            corner_radius=12,
            border_width=1,
            border_color=Theme.BORDER,
        )
        input_frame.pack(fill="x", expand=True, pady=(0, 8))

        # Text input
        self.input_text = ctk.CTkTextbox(
            input_frame,
            height=60,
            fg_color="transparent",
            text_color=Theme.TEXT_PRIMARY,
            font=ctk.CTkFont(size=Theme.FONT_BASE),
            wrap="word",
            border_width=0,
        )
        self.input_text.pack(fill="x", padx=16, pady=(12, 0))
        self.input_text.bind("<Return>", self._on_enter_key)
        self.input_text.bind("<Shift-Return>", lambda e: None)  # Allow shift+enter for newline

        # Bottom row with send button
        bottom_row = ctk.CTkFrame(input_frame, fg_color="transparent")
        bottom_row.pack(fill="x", padx=16, pady=12)

        # File upload button (placeholder)
        attach_btn = ctk.CTkButton(
            bottom_row,
            text="+",
            width=32,
            height=32,
            corner_radius=6,
            fg_color="transparent",
            hover_color=Theme.BG_HOVER,
            text_color=Theme.TEXT_MUTED,
            font=ctk.CTkFont(size=Theme.FONT_XL),
        )
        attach_btn.pack(side="left")

        # Send button
        self.send_btn = ctk.CTkButton(
            bottom_row,
            text="Send",
            width=80,
            height=36,
            corner_radius=8,
            fg_color=Theme.ACCENT,
            hover_color=Theme.ACCENT_HOVER,
            font=ctk.CTkFont(size=Theme.FONT_BASE, weight="bold"),
            command=self._send_message,
        )
        self.send_btn.pack(side="right")

        # Stop button (hidden by default)
        self.stop_btn = ctk.CTkButton(
            bottom_row,
            text="Stop",
            width=80,
            height=36,
            corner_radius=8,
            fg_color=Theme.ERROR,
            hover_color="#dc2626",
            font=ctk.CTkFont(size=Theme.FONT_BASE, weight="bold"),
            command=self._stop_generation,
        )

        # Hint text
        ctk.CTkLabel(
            input_container,
            text="Press Enter to send, Shift+Enter for new line",
            font=ctk.CTkFont(size=Theme.FONT_XS),
            text_color=Theme.TEXT_MUTED,
        ).pack()

        # Load models
        self._load_model_selector()

    def _setup_logging(self):
        """Setup logging to capture messages"""

        class QueueHandler(logging.Handler):
            def __init__(self, log_queue):
                super().__init__()
                self.log_queue = log_queue

            def emit(self, record):
                self.log_queue.put(self.format(record))

        handler = QueueHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    def _setup_keybindings(self):
        """Setup keyboard shortcuts"""
        self.bind("<Control-n>", lambda e: self._new_conversation())
        self.bind("<Control-comma>", lambda e: self._open_settings())

    def _process_logs(self):
        """Process log queue"""
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            # Could display in status or dedicated log panel
        self.after(100, self._process_logs)

    def _initialize_engine(self):
        """Initialize the agent engine"""

        def init():
            try:
                self.ecosystem = AutonomousAgentEcosystem()
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.loop.run_until_complete(self.ecosystem.initialize_agents())

                # Start engine
                self.loop.create_task(self.ecosystem.engine.start_engine())
                self.running = True

                self.after(0, lambda: self._update_status("Online", Theme.SUCCESS))
                self.loop.run_forever()
            except Exception as e:
                logging.error(f"Engine initialization error: {e}")
                self.after(0, lambda: self._update_status("Error", Theme.ERROR))

        threading.Thread(target=init, daemon=True).start()

    def _update_status(self, text: str, color: str):
        """Update status display"""
        self.status_text.configure(text=text)
        self.status_dot.configure(text_color=color)

    def _update_provider_display(self):
        """Update provider label in sidebar"""
        provider = os.getenv("LLM_PROVIDER", "openai")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.provider_label.configure(text=f"{provider.title()} · {model}")

    def _load_model_selector(self):
        """Load models into the header selector"""
        provider = os.getenv("LLM_PROVIDER", "openai")
        models = ModelManager.get_models_for_provider(provider)

        if models:
            model_options = [f"{m.name}" for m in models]
            self.model_selector.configure(values=model_options)

            current = os.getenv("LLM_MODEL", "")
            model_ids = [m.id for m in models]

            if current in model_ids:
                idx = model_ids.index(current)
                self.model_selector_var.set(model_options[idx])
            else:
                self.model_selector_var.set(model_options[0])

    def _on_model_select(self, value):
        """Handle model selection from header"""
        provider = os.getenv("LLM_PROVIDER", "openai")
        models = ModelManager.get_models_for_provider(provider)

        for model in models:
            if model.name == value:
                os.environ["LLM_MODEL"] = model.id
                self._update_provider_display()
                # Reset client
                self.llm_client = None
                break

    def _refresh_conversation_list(self):
        """Refresh the conversation list in sidebar"""
        # Clear existing
        for widget in self.conv_list_frame.winfo_children():
            widget.destroy()

        conversations = self.conversation_store.get_all()
        current_id = self.conversation_store.current_id

        for conv in conversations:
            item = ConversationListItem(
                self.conv_list_frame,
                conv,
                on_select=self._select_conversation,
                on_delete=self._delete_conversation,
                is_active=(conv.id == current_id),
            )
            item.pack(fill="x", pady=2)

    def _new_conversation(self):
        """Create a new conversation"""
        self.conversation_store.create("New Chat")
        self._refresh_conversation_list()
        self._load_current_conversation()

    def _select_conversation(self, conv_id: str):
        """Select a conversation"""
        self.conversation_store.set_current(conv_id)
        self._refresh_conversation_list()
        self._load_current_conversation()

    def _delete_conversation(self, conv_id: str):
        """Delete a conversation"""
        self.conversation_store.delete(conv_id)
        self._refresh_conversation_list()

        # Load another conversation or show welcome
        if not self.conversation_store.current_id:
            conversations = self.conversation_store.get_all()
            if conversations:
                self.conversation_store.set_current(conversations[0].id)
        self._load_current_conversation()

    def _load_current_conversation(self):
        """Load and display current conversation"""
        # Clear messages
        for widget in self.messages_frame.winfo_children():
            widget.destroy()
        self.message_widgets.clear()

        conv = self.conversation_store.get_current()

        if not conv or not conv.messages:
            # Show welcome
            self._show_welcome()
            return

        # Hide welcome, show messages
        for msg in conv.messages:
            self._add_message_widget(msg)

    def _show_welcome(self):
        """Show welcome screen"""
        self.welcome_frame = ctk.CTkFrame(self.messages_frame, fg_color="transparent")
        self.welcome_frame.pack(expand=True, pady=100)

        ctk.CTkLabel(
            self.welcome_frame,
            text="Welcome to ASTRO",
            font=ctk.CTkFont(size=Theme.FONT_3XL, weight="bold"),
            text_color=Theme.TEXT_PRIMARY,
        ).pack()

        ctk.CTkLabel(
            self.welcome_frame,
            text="Your AI assistant with agentic capabilities",
            font=ctk.CTkFont(size=Theme.FONT_LG),
            text_color=Theme.TEXT_MUTED,
        ).pack(pady=(8, 24))

    def _add_message_widget(self, message: Message):
        """Add a message widget to the chat"""
        # Remove welcome if present
        if hasattr(self, "welcome_frame") and self.welcome_frame.winfo_exists():
            self.welcome_frame.destroy()

        bubble = MessageBubble(
            self.messages_frame,
            message,
            on_copy=lambda: None,
            on_regenerate=self._regenerate_response,
        )

        # Align based on role
        if message.role == "user":
            bubble.pack(fill="x", padx=(100, 0), pady=8, anchor="e")
        else:
            bubble.pack(fill="x", padx=(0, 100), pady=8, anchor="w")

        self.message_widgets[message.id] = bubble

        # Scroll to bottom
        self.messages_frame._parent_canvas.yview_moveto(1.0)

    def _set_input(self, text: str):
        """Set input text"""
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", text)
        self.input_text.focus()

    def _on_enter_key(self, event):
        """Handle enter key in input"""
        if not event.state & 0x1:  # Not shift
            self._send_message()
            return "break"
        return None

    def _send_message(self):
        """Send a message"""
        content = self.input_text.get("1.0", "end").strip()
        if not content or self.is_streaming:
            return

        # Clear input
        self.input_text.delete("1.0", "end")

        # Create conversation if needed
        conv = self.conversation_store.get_current()
        if not conv:
            conv = self.conversation_store.create("New Chat")
            self._refresh_conversation_list()

        # Add user message
        user_msg = Message(
            id=str(uuid.uuid4()),
            role="user",
            content=content,
        )
        self.conversation_store.add_message(conv.id, user_msg)
        self._add_message_widget(user_msg)

        # Generate response
        self._generate_response(conv.id, content)

    def _generate_response(self, conv_id: str, user_content: str):
        """Generate AI response"""
        self.is_streaming = True
        self._update_status("Generating...", Theme.WARNING)

        # Show stop button
        self.send_btn.pack_forget()
        self.stop_btn.pack(side="right")

        # Create placeholder message
        assistant_msg = Message(
            id=str(uuid.uuid4()),
            role="assistant",
            content="",
            model=os.getenv("LLM_MODEL", ""),
            is_streaming=True,
        )
        self.current_stream_msg_id = assistant_msg.id
        self.conversation_store.add_message(conv_id, assistant_msg)
        self._add_message_widget(assistant_msg)

        def generate():
            try:
                # Build messages for context
                conv = self.conversation_store.conversations.get(conv_id)
                messages = []

                # System prompt
                messages.append(
                    {
                        "role": "system",
                        "content": """You are ASTRO, an advanced AI assistant with agentic capabilities.
You can help with research, coding, file operations, and more.
Be helpful, accurate, and concise. Use markdown formatting for code blocks.""",
                    }
                )

                # Conversation history (last 10 messages for context)
                if conv:
                    for msg in conv.messages[-10:]:
                        if msg.id != assistant_msg.id:  # Skip current placeholder
                            messages.append({"role": msg.role, "content": msg.content})

                # Initialize client if needed
                provider = os.getenv("LLM_PROVIDER", "openai")
                model = os.getenv("LLM_MODEL", "gpt-4o-mini")

                if not self.llm_client:
                    self.llm_client = LLMFactory.create_unified_client(
                        provider=provider,
                        model=model,
                    )

                if not self.llm_client:
                    raise Exception("Failed to initialize LLM client")

                # Generate response
                full_content = ""

                if provider == "gemini":
                    # Use Gemini streaming
                    response = self.llm_client.chat_sync(messages=messages, stream=True)
                    for chunk in response:
                        if not self.is_streaming:
                            break
                        full_content += chunk.content
                        self.after(0, lambda c=full_content: self._update_stream(conv_id, assistant_msg.id, c))
                else:
                    # Use OpenAI-compatible streaming
                    response = self.llm_client.chat_sync(messages=messages, stream=True)
                    for chunk in response:
                        if not self.is_streaming:
                            break
                        full_content += chunk.content
                        self.after(0, lambda c=full_content: self._update_stream(conv_id, assistant_msg.id, c))

                # Finalize
                self.after(0, lambda: self._finalize_response(conv_id, assistant_msg.id, full_content))

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logging.error(error_msg)
                self.after(0, lambda: self._finalize_response(conv_id, assistant_msg.id, error_msg))

        threading.Thread(target=generate, daemon=True).start()

    def _update_stream(self, conv_id: str, msg_id: str, content: str):
        """Update streaming message"""
        self.conversation_store.update_message(conv_id, msg_id, content)

        # Update widget
        if msg_id in self.message_widgets:
            # Recreate widget with new content
            old_widget = self.message_widgets[msg_id]
            old_widget.destroy()

            conv = self.conversation_store.conversations.get(conv_id)
            if conv:
                for msg in conv.messages:
                    if msg.id == msg_id:
                        msg.content = content
                        self._add_message_widget(msg)
                        break

    def _finalize_response(self, conv_id: str, msg_id: str, content: str):
        """Finalize response after streaming completes"""
        self.is_streaming = False
        self.current_stream_msg_id = None
        self._update_status("Online", Theme.SUCCESS)

        # Show send button
        self.stop_btn.pack_forget()
        self.send_btn.pack(side="right")

        # Update message
        self.conversation_store.update_message(conv_id, msg_id, content)

        # Update widget with final state
        if msg_id in self.message_widgets:
            old_widget = self.message_widgets[msg_id]
            old_widget.destroy()

            conv = self.conversation_store.conversations.get(conv_id)
            if conv:
                for msg in conv.messages:
                    if msg.id == msg_id:
                        msg.content = content
                        msg.is_streaming = False
                        self._add_message_widget(msg)
                        break

        self._refresh_conversation_list()

    def _stop_generation(self):
        """Stop current generation"""
        self.is_streaming = False

    def _regenerate_response(self, message: Message):
        """Regenerate a response"""
        conv = self.conversation_store.get_current()
        if not conv:
            return

        # Find the user message before this assistant message
        user_content = None
        for i, msg in enumerate(conv.messages):
            if msg.id == message.id and i > 0:
                user_content = conv.messages[i - 1].content
                break

        if user_content:
            # Remove the old assistant message
            conv.messages = [m for m in conv.messages if m.id != message.id]

            # Refresh display
            self._load_current_conversation()

            # Generate new response
            self._generate_response(conv.id, user_content)

    def _open_settings(self):
        """Open settings panel"""
        SettingsPanel(self, on_save=self._on_settings_save)

    def _on_settings_save(self):
        """Handle settings save"""
        self._update_provider_display()
        self._load_model_selector()
        self.llm_client = None  # Reset client


if __name__ == "__main__":
    app = ModernChatApp()
    app.mainloop()
