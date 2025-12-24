"""
ASTRO - Modern Terminal User Interface
A rich terminal chat interface with agentic capabilities
"""

import asyncio
import os
import sys
import threading
from typing import Optional, List
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from core.llm_factory import LLMFactory, UnifiedLLMClient
from utils.model_manager import ModelManager


console = Console()


class Theme:
    """Terminal theme colors"""

    PRIMARY = "bright_blue"
    SECONDARY = "dim white"
    SUCCESS = "bright_green"
    WARNING = "bright_yellow"
    ERROR = "bright_red"
    MUTED = "dim"
    ACCENT = "bright_cyan"
    USER = "bright_blue"
    ASSISTANT = "bright_green"


def clear_screen():
    """Clear terminal screen"""
    console.clear()


def print_header():
    """Print application header"""
    header = Text()
    header.append("╭───────────────────────────────────────╮\n", style=Theme.PRIMARY)
    header.append("│", style=Theme.PRIMARY)
    header.append("              ASTRO                    ", style=f"bold {Theme.PRIMARY}")
    header.append("│\n", style=Theme.PRIMARY)
    header.append("│", style=Theme.PRIMARY)
    header.append("     AI Assistant with Agents          ", style=Theme.MUTED)
    header.append("│\n", style=Theme.PRIMARY)
    header.append("╰───────────────────────────────────────╯", style=Theme.PRIMARY)
    console.print(header)
    console.print()


def print_status(provider: str, model: str):
    """Print current status"""
    status_table = Table(show_header=False, box=None, padding=(0, 1))
    status_table.add_column(style=Theme.MUTED)
    status_table.add_column(style="white")
    status_table.add_row("Provider:", provider.title())
    status_table.add_row("Model:", model)
    console.print(Panel(status_table, title="[bold]Status[/bold]", border_style=Theme.MUTED, box=box.ROUNDED))


def print_help():
    """Print help information"""
    help_table = Table(show_header=True, header_style=f"bold {Theme.PRIMARY}", box=box.SIMPLE)
    help_table.add_column("Command", style=Theme.ACCENT)
    help_table.add_column("Description")

    commands = [
        ("/help", "Show this help message"),
        ("/settings", "Configure provider and model"),
        ("/providers", "List available providers"),
        ("/models", "List available models"),
        ("/clear", "Clear the screen"),
        ("/new", "Start a new conversation"),
        ("/exit", "Exit the application"),
    ]

    for cmd, desc in commands:
        help_table.add_row(cmd, desc)

    console.print(Panel(help_table, title="[bold]Commands[/bold]", border_style=Theme.PRIMARY, box=box.ROUNDED))


def print_providers():
    """Print available providers"""
    providers = ModelManager.get_all_providers()

    table = Table(show_header=True, header_style=f"bold {Theme.PRIMARY}", box=box.SIMPLE)
    table.add_column("Provider", style=Theme.ACCENT)
    table.add_column("Description")
    table.add_column("Free", justify="center")
    table.add_column("Status", justify="center")

    for p in providers:
        # Check status
        if p["id"] == "ollama":
            status = ModelManager.check_ollama_status()
            status_str = f"[{Theme.SUCCESS}]Running[/{Theme.SUCCESS}]" if status["running"] else f"[{Theme.WARNING}]Offline[/{Theme.WARNING}]"
        elif p["id"] == "gemini":
            status = ModelManager.check_gemini_status()
            status_str = f"[{Theme.SUCCESS}]Connected[/{Theme.SUCCESS}]" if status.get("available") else f"[{Theme.WARNING}]No Key[/{Theme.WARNING}]"
        else:
            key = os.getenv(p.get("key_env", ""))
            status_str = f"[{Theme.SUCCESS}]Configured[/{Theme.SUCCESS}]" if key else f"[{Theme.WARNING}]No Key[/{Theme.WARNING}]"

        free_str = f"[{Theme.SUCCESS}]Yes[/{Theme.SUCCESS}]" if p.get("free_tier") else "No"

        table.add_row(p["name"], p["description"], free_str, status_str)

    console.print(Panel(table, title="[bold]Available Providers[/bold]", border_style=Theme.PRIMARY, box=box.ROUNDED))


def print_models(provider: str):
    """Print available models for a provider"""
    models = ModelManager.get_models_for_provider(provider)

    table = Table(show_header=True, header_style=f"bold {Theme.PRIMARY}", box=box.SIMPLE)
    table.add_column("Model", style=Theme.ACCENT)
    table.add_column("Description")
    table.add_column("Context", justify="right")
    table.add_column("Features")

    for m in models:
        context = f"{m.context_window:,}" if m.context_window else "-"
        features = []
        if m.supports_vision:
            features.append("Vision")
        if m.supports_tools:
            features.append("Tools")
        features_str = ", ".join(features) if features else "-"

        table.add_row(m.name, m.description or "-", context, features_str)

    console.print(Panel(table, title=f"[bold]Models ({provider.title()})[/bold]", border_style=Theme.PRIMARY, box=box.ROUNDED))


def configure_settings():
    """Interactive settings configuration"""
    console.print(Panel("[bold]Settings Configuration[/bold]", border_style=Theme.PRIMARY, box=box.ROUNDED))

    # Provider selection
    providers = ModelManager.get_all_providers()
    console.print(f"\n[{Theme.PRIMARY}]Available Providers:[/{Theme.PRIMARY}]")
    for i, p in enumerate(providers, 1):
        free_tag = f" [{Theme.SUCCESS}](FREE)[/{Theme.SUCCESS}]" if p.get("free_tier") else ""
        console.print(f"  {i}. {p['name']}{free_tag} - {p['description']}")

    provider_choice = Prompt.ask(
        f"\n[{Theme.PRIMARY}]Select provider[/{Theme.PRIMARY}]",
        choices=[str(i) for i in range(1, len(providers) + 1)],
        default="1",
    )
    selected_provider = providers[int(provider_choice) - 1]
    provider_id = selected_provider["id"]

    # API Key (if required)
    if selected_provider["requires_key"]:
        key_env = selected_provider.get("key_env", "")
        current_key = os.getenv(key_env, "")
        masked_key = f"{current_key[:8]}...{current_key[-4:]}" if len(current_key) > 12 else "(not set)"

        console.print(f"\n[{Theme.MUTED}]Current API key: {masked_key}[/{Theme.MUTED}]")

        new_key = Prompt.ask(
            f"[{Theme.PRIMARY}]Enter API key[/{Theme.PRIMARY}] (press Enter to keep current)",
            default="",
            password=True,
        )

        if new_key:
            os.environ[key_env] = new_key
            console.print(f"[{Theme.SUCCESS}]API key saved[/{Theme.SUCCESS}]")

    # Model selection
    models = ModelManager.get_models_for_provider(provider_id)
    console.print(f"\n[{Theme.PRIMARY}]Available Models:[/{Theme.PRIMARY}]")
    for i, m in enumerate(models[:10], 1):  # Limit to first 10
        console.print(f"  {i}. {m.name} - {m.description}")

    model_choice = Prompt.ask(
        f"\n[{Theme.PRIMARY}]Select model[/{Theme.PRIMARY}]",
        choices=[str(i) for i in range(1, min(len(models) + 1, 11))],
        default="1",
    )
    selected_model = models[int(model_choice) - 1]

    # Save settings
    os.environ["LLM_PROVIDER"] = provider_id
    os.environ["LLM_MODEL"] = selected_model.id

    console.print(f"\n[{Theme.SUCCESS}]Settings saved![/{Theme.SUCCESS}]")
    console.print(f"  Provider: {provider_id.title()}")
    console.print(f"  Model: {selected_model.name}")


def format_message(role: str, content: str) -> Panel:
    """Format a message as a panel"""
    if role == "user":
        return Panel(
            Markdown(content),
            title=f"[bold {Theme.USER}]You[/bold {Theme.USER}]",
            border_style=Theme.USER,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    else:
        return Panel(
            Markdown(content),
            title=f"[bold {Theme.ASSISTANT}]ASTRO[/bold {Theme.ASSISTANT}]",
            border_style=Theme.ASSISTANT,
            box=box.ROUNDED,
            padding=(0, 1),
        )


def get_llm_response(client: UnifiedLLMClient, messages: list, stream: bool = True):
    """Get response from LLM with optional streaming"""
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if stream:
        full_content = ""
        response = client.chat_sync(messages=messages, stream=True)

        with Live(Panel(Text("..."), title=f"[bold {Theme.ASSISTANT}]ASTRO[/bold {Theme.ASSISTANT}]", border_style=Theme.ASSISTANT, box=box.ROUNDED), refresh_per_second=10) as live:
            for chunk in response:
                full_content += chunk.content
                live.update(Panel(Markdown(full_content), title=f"[bold {Theme.ASSISTANT}]ASTRO[/bold {Theme.ASSISTANT}]", border_style=Theme.ASSISTANT, box=box.ROUNDED))

        return full_content
    else:
        with console.status(f"[{Theme.PRIMARY}]Thinking...[/{Theme.PRIMARY}]", spinner="dots"):
            response = client.chat_sync(messages=messages, stream=False)
            return response.content


def run_chat():
    """Main chat loop"""
    clear_screen()
    print_header()

    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    print_status(provider, model)

    console.print(f"\n[{Theme.MUTED}]Type /help for commands, /exit to quit[/{Theme.MUTED}]\n")

    # Initialize conversation
    messages = [
        {
            "role": "system",
            "content": """You are ASTRO, an advanced AI assistant with agentic capabilities.
You can help with research, coding, file operations, and more.
Be helpful, accurate, and concise. Use markdown formatting for code blocks.""",
        }
    ]

    # Initialize client
    client = None

    while True:
        try:
            # Get user input
            user_input = console.input(f"[bold {Theme.USER}]You > [/bold {Theme.USER}]").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()

                if command == "/exit" or command == "/quit":
                    console.print(f"\n[{Theme.MUTED}]Goodbye![/{Theme.MUTED}]\n")
                    break

                elif command == "/help":
                    print_help()
                    continue

                elif command == "/clear":
                    clear_screen()
                    print_header()
                    print_status(os.getenv("LLM_PROVIDER", "openai"), os.getenv("LLM_MODEL", "gpt-4o-mini"))
                    continue

                elif command == "/new":
                    messages = [messages[0]]  # Keep system prompt
                    console.print(f"[{Theme.SUCCESS}]Started new conversation[/{Theme.SUCCESS}]\n")
                    continue

                elif command == "/settings":
                    configure_settings()
                    client = None  # Reset client
                    print_status(os.getenv("LLM_PROVIDER", "openai"), os.getenv("LLM_MODEL", "gpt-4o-mini"))
                    continue

                elif command == "/providers":
                    print_providers()
                    continue

                elif command == "/models":
                    print_models(os.getenv("LLM_PROVIDER", "openai"))
                    continue

                else:
                    console.print(f"[{Theme.WARNING}]Unknown command. Type /help for available commands.[/{Theme.WARNING}]\n")
                    continue

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Print user message
            console.print()

            # Initialize client if needed
            if not client:
                try:
                    client = LLMFactory.create_unified_client(
                        provider=os.getenv("LLM_PROVIDER", "openai"),
                        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                    )
                except Exception as e:
                    console.print(f"[{Theme.ERROR}]Failed to initialize client: {e}[/{Theme.ERROR}]")
                    console.print(f"[{Theme.MUTED}]Use /settings to configure your provider[/{Theme.MUTED}]\n")
                    messages.pop()  # Remove the user message
                    continue

            # Get response
            try:
                response_content = get_llm_response(client, messages, stream=True)
                messages.append({"role": "assistant", "content": response_content})
                console.print()
            except Exception as e:
                console.print(f"[{Theme.ERROR}]Error: {e}[/{Theme.ERROR}]\n")
                messages.pop()  # Remove the user message

        except KeyboardInterrupt:
            console.print(f"\n\n[{Theme.MUTED}]Use /exit to quit[/{Theme.MUTED}]\n")
        except EOFError:
            break


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="ASTRO Terminal Interface")
    parser.add_argument("--provider", type=str, help="LLM provider (openai, gemini, ollama, openrouter)")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--api-key", type=str, help="API key for the provider")
    parser.add_argument("prompt", nargs="?", help="Single prompt to run (non-interactive)")

    args = parser.parse_args()

    # Apply arguments
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.api_key:
        provider = args.provider or os.getenv("LLM_PROVIDER", "openai")
        key_envs = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        if provider in key_envs:
            os.environ[key_envs[provider]] = args.api_key

    # Non-interactive mode
    if args.prompt:
        client = LLMFactory.create_unified_client(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        )

        if not client:
            console.print(f"[{Theme.ERROR}]Failed to initialize client[/{Theme.ERROR}]")
            sys.exit(1)

        messages = [
            {"role": "system", "content": "You are ASTRO, a helpful AI assistant."},
            {"role": "user", "content": args.prompt},
        ]

        response = get_llm_response(client, messages, stream=True)
        sys.exit(0)

    # Interactive mode
    run_chat()


if __name__ == "__main__":
    main()
