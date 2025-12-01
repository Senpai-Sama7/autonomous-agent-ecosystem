#!/usr/bin/env python3
"""
ASTRO Web Server Launcher

This script starts the ASTRO web interface server.
It provides the REST API and WebSocket endpoints for the brutalist web UI.

Usage:
    python run_web.py [--host HOST] [--port PORT] [--reload]

Examples:
    python run_web.py                    # Start on localhost:8000
    python run_web.py --port 3000        # Start on localhost:3000
    python run_web.py --host 0.0.0.0     # Allow external connections
    python run_web.py --reload           # Enable auto-reload for development
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    parser = argparse.ArgumentParser(
        description="ASTRO Web Server - Brutalist Multi-Agent Operations Surface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_web.py                    Start server on http://localhost:8000
  python run_web.py --port 3000        Use custom port
  python run_web.py --host 0.0.0.0     Allow external connections
  python run_web.py --reload           Enable auto-reload for development
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logger = logging.getLogger("ASTRO")
    
    # Print banner
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     █████╗ ███████╗████████╗██████╗  ██████╗              ║
    ║    ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗             ║
    ║    ███████║███████╗   ██║   ██████╔╝██║   ██║             ║
    ║    ██╔══██║╚════██║   ██║   ██╔══██╗██║   ██║             ║
    ║    ██║  ██║███████║   ██║   ██║  ██║╚██████╔╝             ║
    ║    ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝              ║
    ║                                                           ║
    ║    Autonomous Agent Ecosystem                             ║
    ║    Brutalist Web Interface                                ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    logger.info(f"Starting ASTRO Web Server on http://{args.host}:{args.port}")
    logger.info("Press Ctrl+C to stop")
    
    try:
        import uvicorn
        from api.server import create_app
        
        uvicorn.run(
            "api.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="debug" if args.debug else "info",
        )
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install required packages: pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
