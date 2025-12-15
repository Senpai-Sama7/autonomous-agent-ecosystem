"""Command-line interface utilities for Astro."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class CLIOptions:
    """Container for parsed CLI arguments."""

    duration: int = 120
    debug: bool = False
    interactive: bool = False
    llm_provider: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the application."""

    parser = argparse.ArgumentParser(description="Autonomous Agent Ecosystem")
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Duration to run the system in seconds (default: 120)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        help="LLM provider to use (openai, openrouter, ollama)",
    )
    parser.add_argument("--model-name", type=str, help="Model name to use")
    parser.add_argument("--api-key", type=str, help="API key for the LLM provider")
    parser.add_argument("--api-base", type=str, help="Base URL for the LLM API")
    return parser


def parse_arguments(argv: Optional[Sequence[str]] = None) -> CLIOptions:
    """Parse command line arguments into a dataclass instance."""

    args = build_argument_parser().parse_args(argv)
    return CLIOptions(
        duration=args.duration,
        debug=args.debug,
        interactive=args.interactive,
        llm_provider=args.llm_provider,
        model_name=args.model_name,
        api_key=args.api_key,
        api_base=args.api_base,
    )
