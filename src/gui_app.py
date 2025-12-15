"""
ASTRO - Autonomous Agent Ecosystem
Premium Glassmorphic GUI with Onboarding Tutorial
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
import asyncio
import queue
import logging
import os
import webbrowser
from datetime import datetime

from src.bootstrap import build_ecosystem
from src.core.engine import AgentEngine
from src.core.llm_factory import LLMFactory
from src.core.nl_interface import NaturalLanguageInterface
from src.monitoring.monitoring_dashboard import MonitoringDashboard
from src.runtime import AutonomousAgentEcosystem
from src.utils.app_state import AppState
from src.utils.helpers import sanitize_display_text
from src.utils.model_manager import ModelManager

# Configure CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Premium Color Palette
COLORS = {
    # Backgrounds
    "bg_primary": "#09090b",  # Zinc 950
    "bg_secondary": "#18181b",  # Zinc 900
    "bg_card": "#1f1f23",  # Card background
    "bg_hover": "#27272a",  # Hover state
    "bg_input": "#121215",  # Input fields
    # Borders
    "border": "#27272a",  # Subtle border
    "border_light": "#3f3f46",  # Lighter border
    # Accent - Purple gradient feel
    "accent": "#8b5cf6",  # Violet 500
    "accent_hover": "#7c3aed",  # Violet 600
    "accent_light": "#a78bfa",  # Violet 400
    # Status
    "success": "#22c55e",  # Green 500
    "warning": "#eab308",  # Yellow 500
    "error": "#ef4444",  # Red 500
    "info": "#3b82f6",  # Blue 500
    # Text
    "text": "#fafafa",  # Zinc 50
    "text_secondary": "#a1a1aa",  # Zinc 400
    "text_muted": "#71717a",  # Zinc 500
}

# Tutorial Steps - Interactive walkthrough with element targeting
TUTORIAL_STEPS = [
    {
        "title": "Welcome to ASTRO! ",
        "content": "Let me give you a quick tour of your AI assistant team.\n\nClick 'Next' to start the guided walkthrough.",
        "icon": "",
        "target": None,  # No highlight for welcome
        "position": "center",
    },
    {
        "title": "Menu Bar",
        "content": "This is the top menu bar.\n\n'Help' - Opens the help guide\n'Settings' - Configure your AI provider",
        "icon": "",
        "target": "menu_bar",
        "position": "below",
    },
    {
        "title": "Settings - Configure AI",
        "content": "Click 'Settings' to set up your AI:\n\n1. Choose a provider (OpenAI, Ollama, OpenRouter)\n2. Select a model\n3. Enter your API key (if needed)\n\nOllama is FREE and runs locally!",
        "icon": "",
        "target": "settings_btn",
        "position": "below",
    },
    {
        "title": "System Controls",
        "content": "The sidebar contains your main controls:\n\nStatus indicator shows system state\n'Start' activates your AI agents\n'Stop' turns them off when done",
        "icon": "",
        "target": "sidebar_controls",
        "position": "right",
    },
    {
        "title": "Start Button",
        "content": "Click this button to start the AI system.\n\nOnce started, your agents will come online and be ready to help you.",
        "icon": "",
        "target": "start_btn",
        "position": "right",
    },
    {
        "title": "Agent Status",
        "content": "Your AI team members appear here:\n\n Research Agent - Searches the web\n Code Agent - Writes programs\n File Agent - Manages files\n\nGreen dot = Active, Gray = Idle",
        "icon": "",
        "target": "agents_frame",
        "position": "right",
    },
    {
        "title": "Command Input",
        "content": "Type your request here in plain English!\n\nExamples:\n'Research the latest AI news'\n'Write a Python calculator'\n'Create a todo list file'\n\nPress 'Go' or Enter to execute.",
        "icon": "",
        "target": "input_area",
        "position": "below",
    },
    {
        "title": "Reasoning Panel",
        "content": "Watch AI thinking in real-time!\n\nThe 'Reasoning' tab shows:\n- Chain of Thought steps\n- Decision making process\n- Action results\n\nSwitch to 'Logs' for technical details.",
        "icon": "",
        "target": "log_panel",
        "position": "above",
    },
    {
        "title": "Task History",
        "content": "Your completed tasks appear here.\n\nTrack what you've asked and what was accomplished.",
        "icon": "",
        "target": "history_panel",
        "position": "left",
    },
    {
        "title": "You're All Set! ",
        "content": "You now know everything you need!\n\nQuick Start:\n1. Click 'Settings' to configure AI\n2. Click 'Start' to activate agents\n3. Type a command and press 'Go'\n\nHave fun with your AI team!",
        "icon": "",
        "target": None,
        "position": "center",
    },
]


class GlassPanel(ctk.CTkFrame):
    """Glassmorphic panel with subtle border"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            fg_color=COLORS["bg_card"],
            corner_radius=16,
            border_width=1,
            border_color=COLORS["border"],
        )


class AgentCard(ctk.CTkFrame):
    """Minimal agent status indicator"""

    def __init__(self, master, agent_id, capabilities, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            fg_color=COLORS["bg_card"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"],
        )

        # Friendly names with simple icons
        friendly_names = {
            "research_agent_001": ("🔍", "Research", "Searches the web"),
            "code_agent_001": ("⚡", "Code", "Writes & runs code"),
            "filesystem_agent_001": ("📄", "Files", "Manages files"),
            "git_agent_001": ("🔧", "Git", "Version control"),
            "test_agent_001": ("🧪", "Test", "Runs test suites"),
            "analysis_agent_001": ("📊", "Analysis", "Code linting"),
            "knowledge_agent_001": ("🧠", "Knowledge", "Semantic memory"),
        }
        icon, name, desc = friendly_names.get(agent_id, ("●", agent_id, ""))

        # Simple horizontal layout
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="x", padx=16, pady=12)

        # Icon
        ctk.CTkLabel(content, text=icon, font=ctk.CTkFont(size=20)).pack(side="left")

        # Name and description
        text_frame = ctk.CTkFrame(content, fg_color="transparent")
        text_frame.pack(side="left", padx=12, fill="x", expand=True)

        ctk.CTkLabel(
            text_frame,
            text=name,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            text_frame,
            text=desc,
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w")

        # Status dot
        self.status_dot = ctk.CTkLabel(
            content,
            text="●",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_muted"],
        )
        self.status_dot.pack(side="right")

    def set_status(self, status, progress=0):
        colors = {
            "idle": COLORS["text_muted"],
            "active": COLORS["success"],
            "busy": COLORS["warning"],
            "failed": COLORS["error"],
        }
        self.status_dot.configure(
            text_color=colors.get(status.lower(), COLORS["text_muted"])
        )


class TaskItem(ctk.CTkFrame):
    """Minimal task/workflow item"""

    def __init__(self, master, task_name, status, time_str, workflow_id=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color=COLORS["bg_card"], corner_radius=10, height=44)
        self.workflow_id = workflow_id
        self.pack_propagate(False)

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=14, pady=10)

        # Status indicator
        status_colors = {
            "running": COLORS["warning"],
            "completed": COLORS["success"],
            "failed": COLORS["error"],
            "pending": COLORS["text_muted"],
        }

        self.status_dot = ctk.CTkLabel(
            content,
            text="●",
            font=ctk.CTkFont(size=8),
            text_color=status_colors.get(status, COLORS["text_muted"]),
        )
        self.status_dot.pack(side="left")

        # Task name
        ctk.CTkLabel(
            content,
            text=task_name[:40] + "..." if len(task_name) > 40 else task_name,
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text"],
        ).pack(side="left", padx=10)

        # Time
        ctk.CTkLabel(
            content,
            text=time_str,
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_muted"],
        ).pack(side="right")

        self.status_colors = status_colors

    def update_status(self, status):
        self.status_dot.configure(
            text_color=self.status_colors.get(status, COLORS["text_muted"])
        )


# Alias for backward compatibility
WorkflowItem = TaskItem


class TutorialOverlay(ctk.CTkToplevel):
    """Interactive spotlight walkthrough tour"""

    def __init__(self, parent, on_complete=None):
        super().__init__(parent)
        self.parent = parent
        self.on_complete = on_complete
        self.current_step = 0

        # Window setup - transparent overlay
        self.title("")
        self.overrideredirect(True)
        self.attributes("-topmost", True)

        # Wait for parent to be ready
        self.after(100, self._initialize)

    def _initialize(self):
        """Initialize after parent is ready"""
        self.parent.update_idletasks()

        # Get parent geometry
        px, py = self.parent.winfo_rootx(), self.parent.winfo_rooty()
        pw, ph = self.parent.winfo_width(), self.parent.winfo_height()
        self.geometry(f"{pw}x{ph}+{px}+{py}")

        # Create dark overlay background
        self.configure(fg_color="#000000")
        self.attributes("-alpha", 0.85)

        # Create floating tooltip card
        self.tooltip = ctk.CTkFrame(
            self,
            fg_color=COLORS["bg_card"],
            corner_radius=16,
            border_width=2,
            border_color=COLORS["accent"],
        )

        # Build tooltip content
        self._build_tooltip()

        # Show first step
        self._show_step()

        self.lift()
        self.focus_force()

    def _build_tooltip(self):
        """Build the tooltip UI"""
        content = ctk.CTkFrame(self.tooltip, fg_color="transparent")
        content.pack(padx=24, pady=20)

        # Step counter
        self.step_label = ctk.CTkLabel(
            content,
            text="Step 1 of 10",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["accent"],
        )
        self.step_label.pack(anchor="w")

        # Title
        self.title_label = ctk.CTkLabel(
            content,
            text="",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["text"],
        )
        self.title_label.pack(anchor="w", pady=(8, 6))

        # Content
        self.content_label = ctk.CTkLabel(
            content,
            text="",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["text_secondary"],
            wraplength=320,
            justify="left",
        )
        self.content_label.pack(anchor="w", pady=(0, 16))

        # Progress bar
        progress_bg = ctk.CTkFrame(
            content, fg_color=COLORS["border"], height=4, corner_radius=2
        )
        progress_bg.pack(fill="x", pady=(0, 16))

        self.progress_bar = ctk.CTkFrame(
            progress_bg, fg_color=COLORS["accent"], height=4, corner_radius=2
        )
        self.progress_bar.place(relx=0, rely=0, relheight=1, relwidth=0.1)

        # Buttons
        btn_frame = ctk.CTkFrame(content, fg_color="transparent")
        btn_frame.pack(fill="x")

        self.skip_btn = ctk.CTkButton(
            btn_frame,
            text="Skip Tour",
            command=self._skip,
            fg_color="transparent",
            hover_color=COLORS["bg_hover"],
            text_color=COLORS["text_muted"],
            width=90,
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(size=12),
        )
        self.skip_btn.pack(side="left")

        self.next_btn = ctk.CTkButton(
            btn_frame,
            text="Next →",
            command=self._next,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            width=100,
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.next_btn.pack(side="right")

    def _get_widget_for_target(self, target):
        """Get the widget reference for a target name"""
        if target is None:
            return None

        p = self.parent  # Shorthand

        widget_map = {
            "menu_bar": lambda: getattr(p, "menu_bar", None),
            "settings_btn": lambda: getattr(p, "settings_btn", None),
            "sidebar_controls": lambda: (
                p.start_btn.master if hasattr(p, "start_btn") else None
            ),
            "start_btn": lambda: getattr(p, "start_btn", None),
            "agents_frame": lambda: getattr(p, "agents_frame", None),
            "input_area": lambda: (
                p.input_entry.master.master if hasattr(p, "input_entry") else None
            ),
            "log_panel": lambda: p.log_text.master if hasattr(p, "log_text") else None,
            "history_panel": lambda: (
                p.history_frame.master if hasattr(p, "history_frame") else None
            ),
        }

        getter = widget_map.get(target)
        if getter:
            try:
                return getter()
            except:
                return None
        return None

    def _show_step(self):
        """Display the current step"""
        if self.current_step >= len(TUTORIAL_STEPS):
            self._complete()
            return

        step = TUTORIAL_STEPS[self.current_step]

        # Update text
        self.step_label.configure(
            text=f"Step {self.current_step + 1} of {len(TUTORIAL_STEPS)}"
        )
        self.title_label.configure(text=step["title"])
        self.content_label.configure(text=step["content"])

        # Update progress bar
        progress = (self.current_step + 1) / len(TUTORIAL_STEPS)
        self.progress_bar.place(relx=0, rely=0, relheight=1, relwidth=progress)

        # Update button text
        if self.current_step == len(TUTORIAL_STEPS) - 1:
            self.next_btn.configure(text="Finish!")
            self.skip_btn.pack_forget()
        else:
            self.next_btn.configure(text="Next →")

        # Position tooltip near target
        target = step.get("target")
        position = step.get("position", "center")

        widget = self._get_widget_for_target(target)
        self._position_tooltip(widget, position)

    def _position_tooltip(self, widget, position):
        """Position the tooltip relative to target widget"""
        self.tooltip.update_idletasks()
        tw = self.tooltip.winfo_reqwidth()
        th = self.tooltip.winfo_reqheight()

        # Get overlay dimensions
        ow = self.winfo_width()
        oh = self.winfo_height()

        if widget is None or position == "center":
            # Center in overlay
            x = (ow - tw) // 2
            y = (oh - th) // 2
        else:
            try:
                # Get widget position relative to parent
                wx = widget.winfo_rootx() - self.parent.winfo_rootx()
                wy = widget.winfo_rooty() - self.parent.winfo_rooty()
                ww = widget.winfo_width()
                wh = widget.winfo_height()

                if position == "below":
                    x = wx + (ww - tw) // 2
                    y = wy + wh + 20
                elif position == "above":
                    x = wx + (ww - tw) // 2
                    y = wy - th - 20
                elif position == "right":
                    x = wx + ww + 20
                    y = wy + (wh - th) // 2
                elif position == "left":
                    x = wx - tw - 20
                    y = wy + (wh - th) // 2
                else:
                    x = (ow - tw) // 2
                    y = (oh - th) // 2

                # Keep tooltip in bounds
                x = max(20, min(x, ow - tw - 20))
                y = max(20, min(y, oh - th - 20))

            except:
                x = (ow - tw) // 2
                y = (oh - th) // 2

        self.tooltip.place(x=x, y=y)

    def _next(self):
        """Go to next step"""
        self.current_step += 1
        self._show_step()

    def _skip(self):
        """Skip the tutorial"""
        self._complete()

    def _complete(self):
        """Complete the tutorial"""
        AppState().mark_first_run_complete()
        if self.on_complete:
            self.on_complete()
        self.destroy()


class HelpDialog(ctk.CTkToplevel):
    """Help dialog"""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Help")
        self.geometry("550x450")
        self.configure(fg_color=COLORS["bg_primary"])
        self.transient(parent)
        self._build_ui()

    def _build_ui(self):
        content = ctk.CTkScrollableFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=28, pady=28)

        ctk.CTkLabel(
            content,
            text="Help & Guide",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w", pady=(0, 20))

        sections = [
            (
                "🚀 Quick Start",
                "1. Click 'Start' to activate agents\n2. Type your request\n3. Press 'Go' to execute",
            ),
            (
                "💡 Examples",
                "• 'Research quantum computing'\n• 'Write a Python sorting script'\n• 'Save notes to a file'",
            ),
            (
                "⚙️ Providers",
                "OpenAI: Best quality (paid)\nOllama: Free, runs locally\nOpenRouter: Multiple providers",
            ),
        ]

        for title, text in sections:
            frame = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=10)
            frame.pack(fill="x", pady=(0, 10))
            inner = ctk.CTkFrame(frame, fg_color="transparent")
            inner.pack(padx=16, pady=14, fill="x")
            ctk.CTkLabel(
                inner,
                text=title,
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=COLORS["text"],
            ).pack(anchor="w")
            ctk.CTkLabel(
                inner,
                text=text,
                font=ctk.CTkFont(size=12),
                text_color=COLORS["text_secondary"],
                justify="left",
            ).pack(anchor="w", pady=(6, 0))

        ctk.CTkButton(
            content,
            text="🔄 Restart Tutorial",
            command=self._restart_tutorial,
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["bg_hover"],
            text_color=COLORS["text"],
            height=38,
            corner_radius=8,
        ).pack(pady=(16, 0))

    def _restart_tutorial(self):
        AppState().reset_tutorial()
        self.destroy()
        self.master.show_tutorial()


class EnhancedGUI(ctk.CTk):
    """Premium ASTRO GUI with onboarding"""

    def __init__(self):
        super().__init__()
        self.app_state = AppState()

        self.title("ASTRO - AI Assistant Team")
        self.geometry("1300x850")
        self.minsize(1100, 750)
        self.configure(fg_color=COLORS["bg_primary"])

        # Initialize state
        self.ecosystem = build_ecosystem()
        self.log_queue = queue.Queue()
        self.agent_cards = {}
        self.workflow_items = []
        self.running = False
        self.loop = None

        # Setup logging
        self.setup_logging()

        # Layout with menu bar
        self.grid_rowconfigure(0, weight=0)  # Menu
        self.grid_rowconfigure(1, weight=1)  # Content
        self.grid_columnconfigure(0, weight=0)  # Sidebar
        self.grid_columnconfigure(1, weight=1)  # Main

        # Create UI
        self._create_menu_bar()
        self._create_sidebar()
        self._create_main_area()

        # Start loops
        self.after(100, self.process_logs)
        self.after(1000, self.update_status)

        # Check first run
        self.after(500, self._check_first_run)

    def _check_first_run(self):
        if self.app_state.is_first_run():
            self.show_tutorial()

    def show_tutorial(self):
        TutorialOverlay(self)

    def open_help(self):
        HelpDialog(self)

    def _create_menu_bar(self):
        """Top menu bar"""
        self.menu_bar = ctk.CTkFrame(
            self, height=52, fg_color=COLORS["bg_secondary"], corner_radius=0
        )
        self.menu_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.menu_bar.grid_propagate(False)

        # Logo
        ctk.CTkLabel(
            self.menu_bar,
            text="ASTRO",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["text"],
        ).pack(side="left", padx=20)

        # Menu buttons
        right = ctk.CTkFrame(self.menu_bar, fg_color="transparent")
        right.pack(side="right", padx=16)

        self.help_btn = ctk.CTkButton(
            right,
            text="? Help",
            command=self.open_help,
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["bg_hover"],
            text_color=COLORS["text_secondary"],
            width=80,
            height=34,
            corner_radius=8,
            font=ctk.CTkFont(size=13),
        )
        self.help_btn.pack(side="left", padx=4)

        # Settings button - More prominent with accent color
        self.settings_btn = ctk.CTkButton(
            right,
            text="Settings",
            command=self.open_settings,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color="#ffffff",
            width=90,
            height=34,
            corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.settings_btn.pack(side="left", padx=4)

    def _create_sidebar(self):
        """Sidebar with controls"""
        sidebar = ctk.CTkFrame(
            self,
            width=260,
            corner_radius=0,
            fg_color=COLORS["bg_secondary"],
            border_width=1,
            border_color=COLORS["border"],
        )
        sidebar.grid(row=1, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        content = ctk.CTkFrame(sidebar, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)

        # Status
        status_card = ctk.CTkFrame(
            content, fg_color=COLORS["bg_card"], corner_radius=10
        )
        status_card.pack(fill="x", pady=(0, 16))
        status_inner = ctk.CTkFrame(status_card, fg_color="transparent")
        status_inner.pack(padx=14, pady=12, fill="x")

        self.system_status_dot = ctk.CTkLabel(
            status_inner,
            text="●",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"],
        )
        self.system_status_dot.pack(side="left")

        self.system_status_text = ctk.CTkLabel(
            status_inner,
            text="Ready",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["text_secondary"],
        )
        self.system_status_text.pack(side="left", padx=(8, 0))

        # Buttons
        self.start_btn = ctk.CTkButton(
            content,
            text="▶  Start",
            command=self.start_system,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            height=46,
            corner_radius=10,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.start_btn.pack(fill="x", pady=(0, 8))

        self.stop_btn = ctk.CTkButton(
            content,
            text="⏹  Stop",
            command=self.stop_system,
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["bg_hover"],
            height=40,
            corner_radius=10,
            text_color=COLORS["text_secondary"],
            state="disabled",
        )
        self.stop_btn.pack(fill="x", pady=(0, 16))

        # Agents
        ctk.CTkLabel(
            content,
            text="AGENTS",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w", pady=(0, 8))

        self.agents_frame = ctk.CTkScrollableFrame(
            content, fg_color="transparent", scrollbar_button_color=COLORS["border"]
        )
        self.agents_frame.pack(fill="both", expand=True)

    def _create_main_area(self):
        """Main content"""
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=1, column=1, sticky="nsew", padx=20, pady=20)
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # Input card
        input_card = ctk.CTkFrame(
            main,
            fg_color=COLORS["bg_card"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        input_card.grid(row=0, column=0, sticky="ew", pady=(0, 14))

        input_content = ctk.CTkFrame(input_card, fg_color="transparent")
        input_content.pack(fill="x", padx=22, pady=22)

        ctk.CTkLabel(
            input_content,
            text="What would you like me to do?",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w", pady=(0, 4))
        ctk.CTkLabel(
            input_content,
            text="Type a command and press Go",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w", pady=(0, 14))

        # Input row
        input_row = ctk.CTkFrame(input_content, fg_color="transparent")
        input_row.pack(fill="x")
        input_row.grid_columnconfigure(0, weight=1)

        self.input_entry = ctk.CTkEntry(
            input_row,
            placeholder_text="Example: Research the latest AI news and summarize it",
            height=50,
            border_width=1,
            border_color=COLORS["border"],
            fg_color=COLORS["bg_input"],
            font=ctk.CTkFont(size=14),
            text_color=COLORS["text"],
            placeholder_text_color=COLORS["text_muted"],
        )
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        self.input_entry.bind("<Return>", self.handle_input)

        self.send_btn = ctk.CTkButton(
            input_row,
            text="Go",
            width=80,
            height=50,
            command=self.handle_input_btn,
            corner_radius=10,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.send_btn.grid(row=0, column=1)

        # Activity area - Two columns
        activity = ctk.CTkFrame(main, fg_color="transparent")
        activity.grid(row=1, column=0, sticky="nsew")
        activity.grid_columnconfigure(0, weight=2)
        activity.grid_columnconfigure(1, weight=1)
        activity.grid_rowconfigure(0, weight=1)

        # Left panel with tabs (CoT + Logs)
        left_panel = GlassPanel(activity)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # Tab header
        tab_header = ctk.CTkFrame(left_panel, fg_color="transparent")
        tab_header.pack(fill="x", padx=16, pady=(12, 0))

        self.active_tab = ctk.StringVar(value="cot")

        self.cot_tab_btn = ctk.CTkButton(
            tab_header,
            text="Reasoning",
            width=90,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            command=lambda: self._switch_tab("cot"),
        )
        self.cot_tab_btn.pack(side="left", padx=(0, 4))

        self.logs_tab_btn = ctk.CTkButton(
            tab_header,
            text="Logs",
            width=70,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=12),
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["bg_hover"],
            text_color=COLORS["text_secondary"],
            command=lambda: self._switch_tab("logs"),
        )
        self.logs_tab_btn.pack(side="left", padx=(0, 8))

        # Clear button
        ctk.CTkButton(
            tab_header,
            text="Clear",
            command=self.clear_logs,
            fg_color="transparent",
            hover_color=COLORS["bg_hover"],
            width=50,
            height=28,
            corner_radius=6,
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"],
        ).pack(side="right")

        # CoT (Chain of Thought) view - shows reasoning steps
        self.cot_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        self.cot_frame.pack(fill="both", expand=True, padx=16, pady=(8, 16))

        self.cot_text = ctk.CTkTextbox(
            self.cot_frame,
            fg_color=COLORS["bg_input"],
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=12),
            wrap="word",
            corner_radius=8,
        )
        self.cot_text.pack(fill="both", expand=True)
        self.cot_text.insert("end", "Reasoning steps will appear here...\n\n")
        self.cot_text.configure(state="disabled")

        # Logs view - hidden by default
        self.logs_frame = ctk.CTkFrame(left_panel, fg_color="transparent")

        self.log_text = ctk.CTkTextbox(
            self.logs_frame,
            fg_color=COLORS["bg_input"],
            text_color=COLORS["text"],
            font=("SF Mono", 11),
            wrap="word",
            corner_radius=8,
        )
        self.log_text.pack(fill="both", expand=True)

        # Tasks panel (right side)
        tasks_panel = GlassPanel(activity)
        tasks_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        ctk.CTkLabel(
            tasks_panel,
            text="Tasks",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w", padx=20, pady=(16, 8))

        self.history_frame = ctk.CTkScrollableFrame(
            tasks_panel,
            fg_color="transparent",
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["text_muted"],
        )
        self.history_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _switch_tab(self, tab):
        """Switch between CoT and Logs tabs"""
        self.active_tab.set(tab)

        if tab == "cot":
            self.cot_tab_btn.configure(
                fg_color=COLORS["accent"],
                text_color="#ffffff",
                font=ctk.CTkFont(size=12, weight="bold"),
            )
            self.logs_tab_btn.configure(
                fg_color=COLORS["bg_card"],
                text_color=COLORS["text_secondary"],
                font=ctk.CTkFont(size=12),
            )
            self.logs_frame.pack_forget()
            self.cot_frame.pack(fill="both", expand=True, padx=16, pady=(8, 16))
        else:
            self.logs_tab_btn.configure(
                fg_color=COLORS["accent"],
                text_color="#ffffff",
                font=ctk.CTkFont(size=12, weight="bold"),
            )
            self.cot_tab_btn.configure(
                fg_color=COLORS["bg_card"],
                text_color=COLORS["text_secondary"],
                font=ctk.CTkFont(size=12),
            )
            self.cot_frame.pack_forget()
            self.logs_frame.pack(fill="both", expand=True, padx=16, pady=(8, 16))

    def add_reasoning_step(self, step_text, step_type="thought"):
        """Add a reasoning step to the CoT display"""
        # SECURITY: Sanitize step text before display
        safe_text = sanitize_display_text(step_text, max_length=2000)

        self.cot_text.configure(state="normal")

        # Add icon based on type
        icons = {
            "thought": "Thinking",
            "action": "Action",
            "observation": "Result",
            "error": "Error",
        }
        icon = icons.get(step_type, "")

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.cot_text.insert("end", f"[{timestamp}] {icon}\n")
        self.cot_text.insert("end", f"{safe_text}\n\n")
        self.cot_text.see("end")
        self.cot_text.configure(state="disabled")

    def show_error_popup(self, error_message, suggested_fixes=None):
        """Show error popup with actionable fixes"""
        ErrorPopup(self, error_message, suggested_fixes)

    def open_settings(self):
        """Open settings dialog"""
        if (
            not hasattr(self, "settings_window")
            or not self.settings_window.winfo_exists()
        ):
            self.settings_window = SettingsDialog(self)
        else:
            self.settings_window.focus()


class ErrorPopup(ctk.CTkToplevel):
    """Popup dialog for errors with actionable fixes"""

    def __init__(self, parent, error_message, suggested_fixes=None):
        super().__init__(parent)
        self.parent = parent
        self.suggested_fixes = suggested_fixes or []

        self.title("Error")
        self.geometry("450x350")
        self.resizable(False, False)
        self.configure(fg_color=COLORS["bg_primary"])

        self.transient(parent)
        self.grab_set()

        self._build_ui(error_message)

        self.focus_force()
        self.lift()

    def _build_ui(self, error_message):
        """Build error popup UI"""
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=24, pady=24)

        # Error header
        header = ctk.CTkFrame(
            content, fg_color=COLORS["error"], corner_radius=12, height=60
        )
        header.pack(fill="x", pady=(0, 16))
        header.pack_propagate(False)

        header_content = ctk.CTkFrame(header, fg_color="transparent")
        header_content.pack(expand=True, fill="both", padx=16, pady=12)

        ctk.CTkLabel(
            header_content,
            text="! Error Occurred",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#ffffff",
        ).pack(side="left")

        # Error message
        msg_frame = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=10)
        msg_frame.pack(fill="x", pady=(0, 16))

        ctk.CTkLabel(
            msg_frame,
            text=error_message,
            font=ctk.CTkFont(size=13),
            text_color=COLORS["text"],
            wraplength=380,
            justify="left",
        ).pack(padx=16, pady=12, anchor="w")

        # Suggested fixes
        if self.suggested_fixes:
            ctk.CTkLabel(
                content,
                text="Suggested Fixes:",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=COLORS["text"],
            ).pack(anchor="w", pady=(0, 8))

            fixes_frame = ctk.CTkScrollableFrame(
                content, fg_color="transparent", height=100
            )
            fixes_frame.pack(fill="x", pady=(0, 16))

            for i, fix in enumerate(self.suggested_fixes, 1):
                fix_btn = ctk.CTkButton(
                    fixes_frame,
                    text=f"{i}. {fix['label']}",
                    command=lambda f=fix: self._apply_fix(f),
                    fg_color=COLORS["bg_card"],
                    hover_color=COLORS["bg_hover"],
                    text_color=COLORS["text"],
                    anchor="w",
                    height=36,
                    corner_radius=8,
                )
                fix_btn.pack(fill="x", pady=2)

        # Dismiss button
        ctk.CTkButton(
            content,
            text="Dismiss",
            command=self.destroy,
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["bg_hover"],
            text_color=COLORS["text"],
            height=40,
            corner_radius=8,
        ).pack(fill="x")

    def _apply_fix(self, fix):
        """Apply a suggested fix"""
        if "action" in fix and callable(fix["action"]):
            fix["action"]()
        self.destroy()


class SettingsDialog(ctk.CTkToplevel):
    """Intuitive settings with smart model selection"""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Settings")
        self.geometry("500x650")
        self.resizable(False, False)
        self.configure(fg_color=COLORS["bg_primary"])

        self.transient(parent)
        self.after(150, self._setup_grab)

        self.models_cache = {}
        self._build_ui()

        self.focus_force()
        self.lift()

    def _setup_grab(self):
        try:
            if self.winfo_exists() and self.winfo_viewable():
                self.grab_set()
        except Exception:
            pass

    def _build_ui(self):
        """Build intuitive settings UI"""
        # Scrollable content
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=24, pady=24)

        # Title
        ctk.CTkLabel(
            scroll,
            text="Settings",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w", pady=(0, 4))

        ctk.CTkLabel(
            scroll,
            text="Configure your AI provider and model",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w", pady=(0, 24))

        # === STEP 1: Choose Provider ===
        self._create_section(
            scroll,
            "1",
            "Choose your AI provider",
            "Select where your AI models come from",
        )

        provider_frame = ctk.CTkFrame(
            scroll, fg_color=COLORS["bg_card"], corner_radius=12
        )
        provider_frame.pack(fill="x", pady=(0, 20))

        self.provider_var = ctk.StringVar(value=os.getenv("LLM_PROVIDER", "openai"))
        self.provider_var.trace_add("write", self._on_provider_change)

        providers = [
            ("openai", "OpenAI", "GPT-4, GPT-3.5 - Requires API key"),
            ("ollama", "Ollama (Local)", "Free, runs on your computer"),
            ("openrouter", "OpenRouter", "Access many models with one key"),
        ]

        for i, (value, name, desc) in enumerate(providers):
            self._create_provider_option(provider_frame, value, name, desc, i == 0)

        # === STEP 2: Select Model ===
        self._create_section(
            scroll, "2", "Select a model", "Choose which AI model to use"
        )

        self.model_frame = ctk.CTkFrame(
            scroll, fg_color=COLORS["bg_card"], corner_radius=12
        )
        self.model_frame.pack(fill="x", pady=(0, 20))

        self.model_content = ctk.CTkFrame(self.model_frame, fg_color="transparent")
        self.model_content.pack(fill="x", padx=16, pady=16)

        # Model dropdown (will be populated)
        self.model_var = ctk.StringVar(value=os.getenv("LLM_MODEL", "gpt-4o-mini"))
        self.model_dropdown = ctk.CTkOptionMenu(
            self.model_content,
            variable=self.model_var,
            values=["Loading..."],
            fg_color=COLORS["bg_input"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            dropdown_fg_color=COLORS["bg_card"],
            dropdown_hover_color=COLORS["bg_hover"],
            dropdown_text_color=COLORS["text"],
            text_color=COLORS["text"],
            height=44,
            corner_radius=10,
            font=ctk.CTkFont(size=13),
        )
        self.model_dropdown.pack(fill="x")

        # Model info label
        self.model_info = ctk.CTkLabel(
            self.model_content,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"],
        )
        self.model_info.pack(anchor="w", pady=(8, 0))

        # Ollama-specific: Download section (hidden by default)
        self.ollama_section = ctk.CTkFrame(scroll, fg_color="transparent")

        # === STEP 3: API Key (conditional) ===
        self.api_section = ctk.CTkFrame(scroll, fg_color="transparent")
        self.api_section.pack(fill="x")

        self._create_section(
            self.api_section,
            "3",
            "Enter your API key",
            "Required for OpenAI and OpenRouter",
        )

        api_frame = ctk.CTkFrame(
            self.api_section, fg_color=COLORS["bg_card"], corner_radius=12
        )
        api_frame.pack(fill="x", pady=(0, 20))

        api_content = ctk.CTkFrame(api_frame, fg_color="transparent")
        api_content.pack(fill="x", padx=16, pady=16)

        self.api_key_entry = ctk.CTkEntry(
            api_content,
            show="•",
            fg_color=COLORS["bg_input"],
            border_color=COLORS["border"],
            text_color=COLORS["text"],
            placeholder_text_color=COLORS["text_muted"],
            height=44,
            corner_radius=10,
            placeholder_text="Paste your API key here",
        )
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            self.api_key_entry.insert(0, api_key)
        self.api_key_entry.pack(fill="x")

        ctk.CTkLabel(
            api_content,
            text="Your key is stored locally and never shared",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w", pady=(8, 0))

        # === Save Button ===
        ctk.CTkButton(
            scroll,
            text="Save Settings",
            command=self.save_settings,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color=COLORS["text"],
            height=50,
            corner_radius=12,
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(fill="x", pady=(10, 0))

        # Load models for current provider
        self._load_models()

    def _create_section(self, parent, number, title, subtitle):
        """Create a numbered section header"""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=(0, 8))

        # Number badge
        badge = ctk.CTkLabel(
            frame,
            text=number,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["bg_primary"],
            fg_color=COLORS["accent"],
            corner_radius=12,
            width=24,
            height=24,
        )
        badge.pack(side="left")

        text_frame = ctk.CTkFrame(frame, fg_color="transparent")
        text_frame.pack(side="left", padx=(10, 0))

        ctk.CTkLabel(
            text_frame,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            text_frame,
            text=subtitle,
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w")

    def _create_provider_option(self, parent, value, name, desc, is_first):
        """Create a provider radio option"""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(
            fill="x", padx=16, pady=(16 if is_first else 0, 0 if is_first else 16)
        )

        radio = ctk.CTkRadioButton(
            frame,
            text="",
            variable=self.provider_var,
            value=value,
            fg_color=COLORS["accent"],
            border_color=COLORS["border"],
            hover_color=COLORS["accent_hover"],
        )
        radio.pack(side="left")

        text_frame = ctk.CTkFrame(frame, fg_color="transparent")
        text_frame.pack(side="left", padx=(8, 0), fill="x", expand=True)

        ctk.CTkLabel(
            text_frame,
            text=name,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            text_frame,
            text=desc,
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w")

    def _on_provider_change(self, *args):
        """Handle provider change"""
        provider = self.provider_var.get()

        # Show/hide API key section
        if provider == "ollama":
            self.api_section.pack_forget()
            self._show_ollama_section()
        else:
            self.ollama_section.pack_forget()
            self.api_section.pack(fill="x")

        self._load_models()

    def _show_ollama_section(self):
        """Show Ollama-specific options"""
        # Check if Ollama is running
        status = ModelManager.check_ollama_status()

        if not status["running"]:
            self.model_info.configure(
                text="⚠️ Ollama not running. Start it first: ollama serve",
                text_color=COLORS["warning"],
            )

    def _load_models(self):
        """Load models for current provider"""
        provider = self.provider_var.get()

        # Get models
        models = ModelManager.get_models_for_provider(
            provider,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434"),
        )

        if models:
            model_names = [f"{m.name}" for m in models]
            model_ids = [m.id for m in models]
            self.models_cache = {m.name: m for m in models}

            self.model_dropdown.configure(values=model_names)

            # Set current model or first available
            current = os.getenv("LLM_MODEL", "")
            if current in model_ids:
                idx = model_ids.index(current)
                self.model_var.set(model_names[idx])
            else:
                self.model_var.set(model_names[0])

            # Update info
            self._update_model_info()
        else:
            self.model_dropdown.configure(values=["No models available"])
            self.model_var.set("No models available")

            if provider == "ollama":
                self.model_info.configure(
                    text="No models downloaded. Run: ollama pull llama3.2",
                    text_color=COLORS["warning"],
                )

    def _update_model_info(self):
        """Update model info label"""
        model_name = self.model_var.get()
        if model_name in self.models_cache:
            model = self.models_cache[model_name]
            info = model.description
            if model.size:
                info += f" • {model.size}"
            self.model_info.configure(text=info, text_color=COLORS["text_muted"])

    def save_settings(self):
        """Save settings"""
        provider = self.provider_var.get()
        os.environ["LLM_PROVIDER"] = provider

        # Get model ID from name
        model_name = self.model_var.get()
        if model_name in self.models_cache:
            os.environ["LLM_MODEL"] = self.models_cache[model_name].id
        else:
            os.environ["LLM_MODEL"] = model_name

        # API key (only for non-Ollama)
        if provider != "ollama":
            api_key = self.api_key_entry.get()
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key

        # Set base URL for Ollama
        if provider == "ollama":
            os.environ["LLM_BASE_URL"] = "http://localhost:11434/v1"

        # Force re-initialization
        if hasattr(self.parent, "llm_client"):
            delattr(self.parent, "llm_client")
        if hasattr(self.parent, "nl_interface"):
            delattr(self.parent, "nl_interface")

        logging.info(f"✅ Settings saved: {provider} / {os.getenv('LLM_MODEL')}")
        self.destroy()


# Add methods to EnhancedGUI class
EnhancedGUI.setup_logging = lambda self: _setup_logging_impl(self)
EnhancedGUI.process_logs = lambda self: _process_logs_impl(self)
EnhancedGUI.update_status = lambda self: _update_status_impl(self)
EnhancedGUI.start_system = lambda self: _start_system_impl(self)
EnhancedGUI.stop_system = lambda self: _stop_system_impl(self)
EnhancedGUI.clear_logs = lambda self: _clear_logs_impl(self)
EnhancedGUI.run_async_loop = lambda self: _run_async_loop_impl(self)
EnhancedGUI.handle_input_btn = lambda self: _handle_input_btn_impl(self)
EnhancedGUI.handle_input = lambda self, event: _handle_input_impl(self, event)
EnhancedGUI.process_command = lambda self, cmd: _process_command_impl(self, cmd)
EnhancedGUI._process_command_async = lambda self, cmd: _process_command_async_impl(
    self, cmd
)


def _setup_logging_impl(self):
    """Setup logging with queue handler"""

    class QueueHandler(logging.Handler):
        def __init__(self, log_queue):
            super().__init__()
            self.log_queue = log_queue

        def emit(self, record):
            self.log_queue.put(self.format(record))

    handler = QueueHandler(self.log_queue)
    formatter = logging.Formatter(
        "%(asctime)s │ %(levelname)s │ %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)


def _process_logs_impl(self):
    """Process log queue - route to CoT or Logs based on content"""
    # Keywords that indicate reasoning/CoT content
    cot_keywords = [
        "thinking",
        "reasoning",
        "analyzing",
        "considering",
        "step",
        "thought",
        "chain",
        "plan",
        "approach",
        "strategy",
        "decision",
    ]
    error_keywords = ["error", "exception", "failed", "failure", "traceback"]

    while not self.log_queue.empty():
        raw_msg = self.log_queue.get()
        # SECURITY: Sanitize all log messages before display to prevent UI corruption
        msg = sanitize_display_text(raw_msg, max_length=5000)
        msg_lower = msg.lower()

        # Check if it's an error - show popup
        if any(kw in msg_lower for kw in error_keywords) and "ERROR" in msg:
            # Extract error message
            error_msg = msg.split("│")[-1].strip() if "│" in msg else msg

            # Generate suggested fixes based on error type
            fixes = []
            if "api key" in msg_lower or "authentication" in msg_lower:
                fixes.append(
                    {
                        "label": "Open Settings to configure API key",
                        "action": lambda: self.open_settings(),
                    }
                )
            if "connection" in msg_lower or "network" in msg_lower:
                fixes.append(
                    {"label": "Check your internet connection", "action": None}
                )
            if "ollama" in msg_lower:
                fixes.append(
                    {"label": "Start Ollama server (ollama serve)", "action": None}
                )
                fixes.append(
                    {"label": "Pull a model (ollama pull llama3.2)", "action": None}
                )
            if "model" in msg_lower and "not found" in msg_lower:
                fixes.append(
                    {
                        "label": "Open Settings to select a different model",
                        "action": lambda: self.open_settings(),
                    }
                )

            # Show error popup (but don't block, just schedule it)
            self.after(
                100,
                lambda m=error_msg, f=fixes: self.show_error_popup(m, f) if f else None,
            )

            # Also add to CoT as error
            self.add_reasoning_step(error_msg, "error")

        # Check if it's reasoning/CoT content
        elif any(kw in msg_lower for kw in cot_keywords):
            # Extract the message content
            content = msg.split("│")[-1].strip() if "│" in msg else msg

            # Determine step type
            if "action" in msg_lower or "executing" in msg_lower:
                step_type = "action"
            elif (
                "result" in msg_lower
                or "observation" in msg_lower
                or "found" in msg_lower
            ):
                step_type = "observation"
            else:
                step_type = "thought"

            self.add_reasoning_step(content, step_type)

        # All messages go to logs
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    self.after(100, self.process_logs)


def _update_status_impl(self):
    """Update agent status cards and workflow items"""
    if (
        self.running
        and hasattr(self, "ecosystem")
        and hasattr(self.ecosystem, "engine")
    ):
        # Update Agent Cards
        for agent_id, card in self.agent_cards.items():
            if agent_id in self.ecosystem.engine.agent_status:
                status = self.ecosystem.engine.agent_status[agent_id].value
                progress = len(self.ecosystem.engine.active_tasks) / 10.0
                card.set_status(status, min(progress, 1.0))

        # Update Workflow Items
        for item in self.workflow_items:
            if item.workflow_id and item.workflow_id in self.ecosystem.engine.workflows:
                workflow = self.ecosystem.engine.workflows[item.workflow_id]
                all_tasks = workflow.tasks
                completed = sum(
                    1
                    for t in all_tasks
                    if t.task_id in self.ecosystem.engine.completed_tasks
                )
                failed = sum(
                    1
                    for t in all_tasks
                    if t.task_id in self.ecosystem.engine.failed_tasks
                )

                if failed > 0:
                    item.update_status("failed")
                elif completed == len(all_tasks):
                    item.update_status("completed")
                else:
                    item.update_status("running")

    self.after(1000, self.update_status)


def _start_system_impl(self):
    """Start the agent system"""
    if not self.running:
        self.running = True
        self.system_status_dot.configure(text_color=COLORS["success"])
        self.system_status_text.configure(
            text="System Online", text_color=COLORS["success"]
        )

        # Disable start button, enable stop
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        # Start async loop
        threading.Thread(target=self.run_async_loop, daemon=True).start()

        logging.info("🚀 ASTRO system started successfully!")
        logging.info("💡 Type a command above to get started")


def _stop_system_impl(self):
    """Stop the agent system"""
    self.running = False
    self.system_status_dot.configure(text_color=COLORS["text_muted"])
    self.system_status_text.configure(
        text="System Offline", text_color=COLORS["text_muted"]
    )

    # Enable start button, disable stop
    self.start_btn.configure(state="normal")
    self.stop_btn.configure(state="disabled")

    logging.info("⏹️ System stopped")


def _clear_logs_impl(self):
    """Clear both CoT and log displays"""
    # Clear logs
    self.log_text.delete("1.0", "end")

    # Clear CoT
    self.cot_text.configure(state="normal")
    self.cot_text.delete("1.0", "end")
    self.cot_text.insert("end", "Reasoning steps will appear here...\n\n")
    self.cot_text.configure(state="disabled")

    logging.info("Cleared")


def _run_async_loop_impl(self):
    """Run asyncio loop in background thread"""
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)

    try:
        # Initialize agents
        self.loop.run_until_complete(self.ecosystem.initialize_agents())

        # Create agent cards in main thread
        def create_cards():
            for agent_id, agent in self.ecosystem.agents.items():
                caps = []
                if hasattr(agent, "capabilities"):
                    caps = [c.value for c in agent.capabilities]

                card = AgentCard(self.agents_frame, agent_id, caps)
                card.pack(fill="x", pady=4)
                self.agent_cards[agent_id] = card

        self.after(0, create_cards)

        # Start engine
        self.loop.create_task(self.ecosystem.engine.start_engine())
        self.loop.create_task(self.ecosystem.dashboard.start_monitoring())

        self.loop.run_forever()
    except Exception as e:
        logging.error(f"❌ Engine error: {e}")


def _handle_input_btn_impl(self):
    self.handle_input(None)


def _handle_input_impl(self, event):
    """Handle user input"""
    user_input = self.input_entry.get().strip()
    if not user_input:
        return

    self.input_entry.delete(0, "end")

    if not self.running:
        messagebox.showinfo("Start First", "Click the Start button to begin!")
        return

    # Add to history
    time_str = datetime.now().strftime("%H:%M")
    task_item = TaskItem(self.history_frame, user_input, "running", time_str)
    task_item.pack(fill="x", pady=4)
    self.workflow_items.append(task_item)

    # Process command
    logging.info(f"📝 {user_input}")
    threading.Thread(
        target=self.process_command, args=(user_input,), daemon=True
    ).start()


def _process_command_impl(self, command):
    """Process command asynchronously"""
    if self.loop:
        asyncio.run_coroutine_threadsafe(
            _process_command_async_impl(self, command), self.loop
        )


async def _process_command_async_impl(self, command):
    """Process command with NL interface"""
    # Initialize client if needed
    if not hasattr(self, "llm_client"):
        self.llm_client = LLMFactory.create_client(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )

    # Initialize NL Interface if needed
    if not hasattr(self, "nl_interface"):
        self.nl_interface = NaturalLanguageInterface(
            engine=self.ecosystem.engine,
            llm_client=self.llm_client,
            model_name=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        )

    try:
        workflow_id = await self.nl_interface.process_request(command)
        logging.info(f"✅ Workflow created: {workflow_id}")
    except Exception as e:
        logging.error(f"❌ Error: {e}")


if __name__ == "__main__":
    app = EnhancedGUI()
    app.mainloop()
