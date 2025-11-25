"""
Autonomous Agent Ecosystem - Modern GUI
Built with CustomTkinter for a sleek, modern look with clay-morphism elements.
"""
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import threading
import asyncio
import queue
import logging
import sys
import os
from PIL import Image, ImageTk, ImageDraw

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.engine import AgentEngine
from core.nl_interface import NaturalLanguageInterface
from monitoring.monitoring_dashboard import MonitoringDashboard
from main import AutonomousAgentEcosystem

# Configure CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ClayFrame(ctk.CTkFrame):
    """Custom frame with clay-morphism style (soft shadows, rounded corners)"""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="#2b2b2b", corner_radius=20, border_width=0)
        
        # Add subtle inner shadow/highlight effect (simulated with borders for now)
        # In a real claymorphism implementation, we'd use canvas drawing for complex shadows
        self.configure(border_color="#3a3a3a", border_width=2)

class ModernGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Autonomous Agent Ecosystem")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
        # Initialize Ecosystem
        self.ecosystem = AutonomousAgentEcosystem()
        self.log_queue = queue.Queue()
        self.setup_logging()
        
        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ClayFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(4, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="AGENTS", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.status_label = ctk.CTkLabel(self.sidebar, text="● System Offline", text_color="gray")
        self.status_label.grid(row=1, column=0, padx=20, pady=10)

        # Sidebar Buttons
        self.btn_dashboard = self.create_sidebar_button("Dashboard", 2)
        self.btn_agents = self.create_sidebar_button("Agents", 3)
        self.btn_settings = self.create_sidebar_button("Settings", 4)

        # Main Content Area
        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_area.grid_columnconfigure(0, weight=1)
        self.main_area.grid_rowconfigure(1, weight=1)

        # Header / Chat Input
        self.create_chat_interface()
        
        # Log/Output Area
        self.create_log_area()

        # Start background loop
        self.running = False
        self.after(100, self.process_logs)

    def create_sidebar_button(self, text, row):
        btn = ctk.CTkButton(self.sidebar, text=text, fg_color="transparent", 
                           hover_color="#404040", anchor="w", height=40)
        btn.grid(row=row, column=0, padx=20, pady=5, sticky="ew")
        return btn

    def create_chat_interface(self):
        self.chat_frame = ClayFrame(self.main_area)
        self.chat_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        self.chat_frame.grid_columnconfigure(0, weight=1)

        self.input_entry = ctk.CTkEntry(self.chat_frame, placeholder_text="Ask the agents to do something...",
                                      height=50, border_width=0, fg_color="#333333")
        self.input_entry.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        self.input_entry.bind("<Return>", self.handle_input)

        self.send_btn = ctk.CTkButton(self.chat_frame, text="Execute", width=100, height=50,
                                     command=self.handle_input_btn, corner_radius=15,
                                     fg_color="#4a90e2", hover_color="#357abd")
        self.send_btn.grid(row=0, column=1, padx=(0, 20), pady=20)

    def create_log_area(self):
        self.log_frame = ClayFrame(self.main_area)
        self.log_frame.grid(row=1, column=0, sticky="nsew")
        
        self.log_label = ctk.CTkLabel(self.log_frame, text="System Logs & Activity", 
                                     font=ctk.CTkFont(size=14, weight="bold"))
        self.log_label.pack(padx=20, pady=10, anchor="w")

        self.log_text = ctk.CTkTextbox(self.log_frame, fg_color="#1e1e1e", text_color="#d1d1d1",
                                      font=("Consolas", 12))
        self.log_text.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    def setup_logging(self):
        class QueueHandler(logging.Handler):
            def __init__(self, log_queue):
                super().__init__()
                self.log_queue = log_queue

            def emit(self, record):
                self.log_queue.put(self.format(record))

        handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Attach to root logger
        logging.getLogger().addHandler(handler)

    def process_logs(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
        self.after(100, self.process_logs)

    def handle_input_btn(self):
        self.handle_input(None)

    def handle_input(self, event):
        user_input = self.input_entry.get()
        if not user_input:
            return
            
        self.input_entry.delete(0, "end")
        self.log_text.insert("end", f"\n> USER: {user_input}\n")
        
        if not self.running:
            self.start_system()
            
        # Process in background
        threading.Thread(target=self.process_command, args=(user_input,), daemon=True).start()

    def start_system(self):
        if not self.running:
            self.running = True
            self.status_label.configure(text="● System Online", text_color="#4cd964")
            
            # Start asyncio loop in separate thread
            threading.Thread(target=self.run_async_loop, daemon=True).start()

    def run_async_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Initialize agents
        self.loop.run_until_complete(self.ecosystem.initialize_agents())
        
        # Start engine
        self.loop.create_task(self.ecosystem.engine.start_engine())
        self.loop.create_task(self.ecosystem.dashboard.start_monitoring())
        
        self.loop.run_forever()

    def process_command(self, command):
        # This needs to interact with the running async loop
        if hasattr(self, 'loop'):
            asyncio.run_coroutine_threadsafe(self._process_command_async(command), self.loop)

    async def _process_command_async(self, command):
        # Use NL Interface
        # Note: We need to instantiate NL interface with a client if available
        # For now, we'll use a simple placeholder or the one from main if we refactor
        # Creating a temporary one for this demo
        from core.nl_interface import NaturalLanguageInterface
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except:
            client = None
            
        nl = NaturalLanguageInterface(self.ecosystem.engine, client)
        try:
            workflow_id = await nl.process_request(command)
            logging.info(f"Workflow started: {workflow_id}")
        except Exception as e:
            logging.error(f"Error processing command: {e}")

if __name__ == "__main__":
    app = ModernGUI()
    app.mainloop()
