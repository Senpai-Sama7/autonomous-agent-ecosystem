"""
Autonomous Agent Ecosystem - Enhanced Modern GUI
Built with CustomTkinter for intuitive UX and premium aesthetics.
"""
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import asyncio
import queue
import logging
import sys
import os
from datetime import datetime
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.engine import AgentEngine
from core.nl_interface import NaturalLanguageInterface
from monitoring.monitoring_dashboard import MonitoringDashboard
from main import AutonomousAgentEcosystem

# Configure CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AgentCard(ctk.CTkFrame):
    """Visual card for displaying agent status"""
    def __init__(self, master, agent_id, capabilities, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="#2b2b2b", corner_radius=15, border_width=2, border_color="#3a3a3a")
        
        # Agent name
        self.name_label = ctk.CTkLabel(self, text=agent_id, font=ctk.CTkFont(size=14, weight="bold"))
        self.name_label.pack(padx=15, pady=(10, 5), anchor="w")
        
        # Status indicator
        self.status_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.status_frame.pack(padx=15, pady=5, anchor="w", fill="x")
        
        self.status_dot = ctk.CTkLabel(self.status_frame, text="●", text_color="gray", font=ctk.CTkFont(size=20))
        self.status_dot.pack(side="left")
        
        self.status_label = ctk.CTkLabel(self.status_frame, text="Idle", text_color="gray")
        self.status_label.pack(side="left", padx=5)
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(self, height=8, corner_radius=4)
        self.progress.pack(padx=15, pady=5, fill="x")
        self.progress.set(0)
        
        # Capabilities
        caps_text = ", ".join(capabilities[:2]) + "..." if len(capabilities) > 2 else ", ".join(capabilities)
        self.caps_label = ctk.CTkLabel(self, text=caps_text, text_color="#888", font=ctk.CTkFont(size=10))
        self.caps_label.pack(padx=15, pady=(0, 10), anchor="w")
        
    def set_status(self, status, progress=0):
        colors = {
            "idle": ("gray", "Idle"),
            "active": ("#4cd964", "Active"),
            "busy": ("#ff9500", "Busy"),
            "failed": ("#ff3b30", "Failed")
        }
        color, text = colors.get(status.lower(), ("gray", status))
        self.status_dot.configure(text_color=color)
        self.status_label.configure(text=text)
        self.progress.set(progress)

class WorkflowItem(ctk.CTkFrame):
    """Visual item for workflow in history"""
    def __init__(self, master, workflow_name, status, time_str, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="#1e1e1e", corner_radius=10, height=60)
        
        # Time
        time_label = ctk.CTkLabel(self, text=time_str, text_color="#666", font=ctk.CTkFont(size=10))
        time_label.pack(side="left", padx=10)
        
        # Name
        name_label = ctk.CTkLabel(self, text=workflow_name, font=ctk.CTkFont(size=12))
        name_label.pack(side="left", padx=5, fill="x", expand=True)
        
        # Status badge
        status_colors = {
            "running": "#ff9500",
            "completed": "#4cd964",
            "failed": "#ff3b30"
        }
        badge = ctk.CTkLabel(self, text=status.upper(), text_color=status_colors.get(status, "gray"),
                           font=ctk.CTkFont(size=10, weight="bold"))
        badge.pack(side="right", padx=10)

class EnhancedGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Autonomous Agent Ecosystem")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        
        # Initialize state
        self.ecosystem = AutonomousAgentEcosystem()
        self.log_queue = queue.Queue()
        self.agent_cards = {}
        self.workflow_items = []
        self.setup_logging()
        
        # Main layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create UI
        self.create_sidebar()
        self.create_main_content()
        
        # Start processing
        self.running = False
        self.after(100, self.process_logs)
        self.after(1000, self.update_status)

    def create_sidebar(self):
        """Enhanced sidebar with better organization"""
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color="#1e1e1e")
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(6, weight=1)
        
        # Logo/Title
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="ew")
        
        title = ctk.CTkLabel(logo_frame, text="AGENT", font=ctk.CTkFont(size=28, weight="bold"))
        title.pack(side="left")
        
        subtitle = ctk.CTkLabel(logo_frame, text="ECO", font=ctk.CTkFont(size=28, weight="bold"), 
                               text_color="#4a90e2")
        subtitle.pack(side="left")
        
        # System status
        self.status_frame = ctk.CTkFrame(self.sidebar, fg_color="#2b2b2b", corner_radius=10)
        self.status_frame.grid(row=1, column=0, padx=20, pady=15, sticky="ew")
        
        self.system_status_dot = ctk.CTkLabel(self.status_frame, text="●", text_color="gray", 
                                             font=ctk.CTkFont(size=16))
        self.system_status_dot.pack(side="left", padx=10, pady=10)
        
        self.system_status_text = ctk.CTkLabel(self.status_frame, text="System Offline", text_color="gray")
        self.system_status_text.pack(side="left", pady=10)
        
        # Quick Actions
        actions_label = ctk.CTkLabel(self.sidebar, text="QUICK ACTIONS", 
                                     font=ctk.CTkFont(size=12, weight="bold"), text_color="#666")
        actions_label.grid(row=2, column=0, padx=20, pady=(10, 5), sticky="w")
        
        self.start_btn = self.create_action_button("🚀 Start System", self.start_system, 3, "#4cd964")
        self.stop_btn = self.create_action_button("⏸️ Stop System", self.stop_system, 4, "#ff3b30")
        self.clear_btn = self.create_action_button("🗑️ Clear Logs", self.clear_logs, 5, "#666")
        
        # Agent Cards Section
        agents_label = ctk.CTkLabel(self.sidebar, text="AGENTS", 
                                    font=ctk.CTkFont(size=12, weight="bold"), text_color="#666")
        agents_label.grid(row=6, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Scrollable agent cards
        self.agents_frame = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.agents_frame.grid(row=7, column=0, padx=20, pady=(0, 20), sticky="nsew")
        
    def create_action_button(self, text, command, row, color="#4a90e2"):
        btn = ctk.CTkButton(self.sidebar, text=text, command=command, 
                          fg_color=color, hover_color=self.darken_color(color),
                          height=40, corner_radius=10, anchor="w", 
                          font=ctk.CTkFont(size=13))
        btn.grid(row=row, column=0, padx=20, pady=5, sticky="ew")
        return btn
        
    def darken_color(self, hex_color):
        # Simple color darkening
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r, g, b = max(0, r-30), max(0, g-30), max(0, b-30)
        return f"#{r:02x}{g:02x}{b:02x}"

    def create_main_content(self):
        """Enhanced main content area"""
        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_area.grid_columnconfigure(0, weight=2)
        self.main_area.grid_columnconfigure(1, weight=1)
        self.main_area.grid_rowconfigure(1, weight=1)

        # Top: Chat Interface
        self.create_enhanced_chat()
        
        # Bottom Left: Logs
        self.create_enhanced_logs()
        
        # Bottom Right: Workflow History
        self.create_workflow_history()

    def create_enhanced_chat(self):
        """Enhanced chat interface with better UX"""
        chat_container = ctk.CTkFrame(self.main_area, fg_color="#2b2b2b", corner_radius=20, 
                                     border_width=2, border_color="#3a3a3a")
        chat_container.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 15))
        chat_container.grid_columnconfigure(0, weight=1)
        
        # Header
        header = ctk.CTkLabel(chat_container, text="💬 Ask Your Agents", 
                            font=ctk.CTkFont(size=16, weight="bold"))
        header.grid(row=0, column=0, padx=25, pady=(15, 5), sticky="w")
        
        # Help text
        help_text = ctk.CTkLabel(chat_container, 
                                text="Try: 'Research quantum computing and save a summary to file'",
                                text_color="#666", font=ctk.CTkFont(size=11))
        help_text.grid(row=1, column=0, padx=25, pady=(0, 10), sticky="w")
        
        # Input area
        input_frame = ctk.CTkFrame(chat_container, fg_color="transparent")
        input_frame.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)
        
        self.input_entry = ctk.CTkEntry(input_frame, placeholder_text="Type your command here...",
                                      height=55, border_width=0, fg_color="#1e1e1e",
                                      font=ctk.CTkFont(size=14))
        self.input_entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.input_entry.bind("<Return>", self.handle_input)
        
        self.send_btn = ctk.CTkButton(input_frame, text="Execute ▶", width=120, height=55,
                                     command=self.handle_input_btn, corner_radius=12,
                                     fg_color="#4a90e2", hover_color="#357abd",
                                     font=ctk.CTkFont(size=14, weight="bold"))
        self.send_btn.grid(row=0, column=1)

    def create_enhanced_logs(self):
        """Enhanced log area with better readability"""
        log_container = ctk.CTkFrame(self.main_area, fg_color="#2b2b2b", corner_radius=20,
                                    border_width=2, border_color="#3a3a3a")
        log_container.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        
        # Header
        header_frame = ctk.CTkFrame(log_container, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(15, 10))
        
        title = ctk.CTkLabel(header_frame, text="📡 System Activity", 
                           font=ctk.CTkFont(size=14, weight="bold"))
        title.pack(side="left")
        
        # Auto-scroll toggle
        self.autoscroll_var = tk.BooleanVar(value=True)
        autoscroll_check = ctk.CTkCheckBox(header_frame, text="Auto-scroll", variable=self.autoscroll_var,
                                          font=ctk.CTkFont(size=11))
        autoscroll_check.pack(side="right")
        
        # Log text with better styling
        self.log_text = ctk.CTkTextbox(log_container, fg_color="#1e1e1e", text_color="#d1d1d1",
                                      font=("Consolas", 11), wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Configure tags for colored logs
        self.log_text.tag_config("INFO", foreground="#4cd964")
        self.log_text.tag_config("WARNING", foreground="#ff9500")
        self.log_text.tag_config("ERROR", foreground="#ff3b30")

    def create_workflow_history(self):
        """Workflow history panel"""
        history_container = ctk.CTkFrame(self.main_area, fg_color="#2b2b2b", corner_radius=20,
                                        border_width=2, border_color="#3a3a3a")
        history_container.grid(row=1, column=1, sticky="nsew")
        
        # Header
        title = ctk.CTkLabel(history_container, text="📜 Workflow History", 
                           font=ctk.CTkFont(size=14, weight="bold"))
        title.pack(padx=20, pady=(15, 10), anchor="w")
        
        # Scrollable list
        self.history_frame = ctk.CTkScrollableFrame(history_container, fg_color="transparent")
        self.history_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def setup_logging(self):
        """Setup logging with queue"""
        class QueueHandler(logging.Handler):
            def __init__(self, log_queue):
                super().__init__()
                self.log_queue = log_queue

            def emit(self, record):
                self.log_queue.put(self.format(record))

        handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                     datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

    def process_logs(self):
        """Process log queue with color coding"""
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            
            # Determine tag based on level
            tag = "INFO"
            if "WARNING" in msg:
                tag = "WARNING"
            elif "ERROR" in msg:
                tag = "ERROR"
                
            self.log_text.insert("end", msg + "\n", tag)
            
            if self.autoscroll_var.get():
                self.log_text.see("end")
                
        self.after(100, self.process_logs)

    def update_status(self):
        """Update agent status cards periodically"""
        if self.running and hasattr(self, 'ecosystem') and hasattr(self.ecosystem, 'engine'):
            for agent_id, card in self.agent_cards.items():
                if agent_id in self.ecosystem.engine.agent_status:
                    status = self.ecosystem.engine.agent_status[agent_id].value
                    progress = len(self.ecosystem.engine.active_tasks) / 10.0  # Simple metric
                    card.set_status(status, min(progress, 1.0))
                    
        self.after(1000, self.update_status)

    def start_system(self):
        """Start the agent system"""
        if not self.running:
            self.running = True
            self.system_status_dot.configure(text_color="#4cd964")
            self.system_status_text.configure(text="System Online", text_color="#4cd964")
            
            # Start async loop
            threading.Thread(target=self.run_async_loop, daemon=True).start()
            
            logging.info("🚀 System started successfully")

    def stop_system(self):
        """Stop the agent system"""
        self.running = False
        self.system_status_dot.configure(text_color="gray")
        self.system_status_text.configure(text="System Offline", text_color="gray")
        logging.info("⏸️ System stopped")

    def clear_logs(self):
        """Clear log display"""
        self.log_text.delete("1.0", "end")
        logging.info("Logs cleared")

    def run_async_loop(self):
        """Run asyncio loop in thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Initialize agents
        self.loop.run_until_complete(self.ecosystem.initialize_agents())
        
        # Create agent cards
        for agent_id, agent in self.ecosystem.agents.items():
            caps = []
            if hasattr(agent, 'capabilities'):
                caps = [c.value for c in agent.capabilities]
            
            card = AgentCard(self.agents_frame, agent_id, caps)
            card.pack(fill="x", pady=5)
            self.agent_cards[agent_id] = card
        
        # Start engine
        self.loop.create_task(self.ecosystem.engine.start_engine())
        self.loop.create_task(self.ecosystem.dashboard.start_monitoring())
        
        self.loop.run_forever()

    def handle_input_btn(self):
        self.handle_input(None)

    def handle_input(self, event):
        """Handle user input"""
        user_input = self.input_entry.get().strip()
        if not user_input:
            return
            
        self.input_entry.delete(0, "end")
        
        if not self.running:
            messagebox.showwarning("System Offline", "Please start the system first!")
            return
            
        # Add to history
        time_str = datetime.now().strftime("%H:%M")
        workflow_item = WorkflowItem(self.history_frame, user_input[:30] + "...", "running", time_str)
        workflow_item.pack(fill="x", pady=3)
        self.workflow_items.append(workflow_item)
        
        # Process command
        logging.info(f"💬 USER: {user_input}")
        threading.Thread(target=self.process_command, args=(user_input,), daemon=True).start()

    def process_command(self, command):
        """Process command asynchronously"""
        if hasattr(self, 'loop'):
            asyncio.run_coroutine_threadsafe(self._process_command_async(command), self.loop)

    async def _process_command_async(self, command):
        """Process command with NL interface"""
        from core.nl_interface import NaturalLanguageInterface
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except:
            client = None
            
        nl = NaturalLanguageInterface(self.ecosystem.engine, client)
        try:
            workflow_id = await nl.process_request(command)
            logging.info(f"✅ Workflow created: {workflow_id}")
        except Exception as e:
            logging.error(f"❌ Error: {e}")

if __name__ == "__main__":
    app = EnhancedGUI()
    app.mainloop()
