#!/usr/bin/env python3
"""Initialize new agents into ASTRO ecosystem."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.engine import AgentEngine, AgentConfig
from core.agent_registry import initialize_agents
from core.agent_tools import AgentToolkit
from utils.logger import configure_logging, get_logger

logger = get_logger("AgentInitializer")


async def main():
    """Initialize all agents."""
    configure_logging("INFO")
    
    logger.info("🚀 Initializing ASTRO Agent Ecosystem...")
    
    # Create engine
    engine = AgentEngine()
    
    # Initialize new agents
    agents = await initialize_agents(engine)
    
    # Create toolkit
    toolkit = AgentToolkit(agents)
    
    logger.info("✅ Agent initialization complete!")
    logger.info(f"📦 Registered agents: {list(agents.keys())}")
    
    # Quick test
    logger.info("\n🧪 Running quick tests...")
    
    # Test git agent
    try:
        result = await toolkit.git_ops("status")
        logger.info(f"✓ Git Agent: {result['success']}")
    except Exception as e:
        logger.warning(f"✗ Git Agent test: {e}")
    
    # Test knowledge agent
    try:
        await toolkit.knowledge_manager("save", "init_test", "Agents initialized successfully")
        result = await toolkit.knowledge_manager("retrieve", "init_test")
        logger.info(f"✓ Knowledge Agent: {result['success']}")
    except Exception as e:
        logger.warning(f"✗ Knowledge Agent test: {e}")
    
    logger.info("\n✨ All agents ready for use!")


if __name__ == "__main__":
    asyncio.run(main())
