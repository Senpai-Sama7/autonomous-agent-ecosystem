"""Agent registry and initialization for ASTRO ecosystem."""

import logging
from typing import Dict, Any
from agents.git_agent import GitAgent
from agents.test_agent import TestAgent
from agents.analysis_agent import AnalysisAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.base_agent import AgentCapability
from core.engine import AgentConfig

logger = logging.getLogger(__name__)


async def initialize_agents(engine: Any) -> Dict[str, Any]:
    """Initialize and register all agents with the engine."""
    
    agents = {}
    
    # Git Agent
    git_agent = GitAgent("git_agent_001", {})
    git_config = AgentConfig(
        agent_id="git_agent_001",
        capabilities=["version_control", "diff_analysis"],
        max_concurrent_tasks=2,
    )
    await engine.register_agent(git_config, git_agent)
    agents["git_agent_001"] = git_agent
    logger.info("✅ Git Agent registered")
    
    # Test Agent
    test_agent = TestAgent("test_agent_001", {})
    test_config = AgentConfig(
        agent_id="test_agent_001",
        capabilities=["test_execution", "quality_assurance"],
        max_concurrent_tasks=3,
    )
    await engine.register_agent(test_config, test_agent)
    agents["test_agent_001"] = test_agent
    logger.info("✅ Test Agent registered")
    
    # Analysis Agent
    analysis_agent = AnalysisAgent("analysis_agent_001", {})
    analysis_config = AgentConfig(
        agent_id="analysis_agent_001",
        capabilities=["code_analysis", "linting"],
        max_concurrent_tasks=2,
    )
    await engine.register_agent(analysis_config, analysis_agent)
    agents["analysis_agent_001"] = analysis_agent
    logger.info("✅ Analysis Agent registered")
    
    # Knowledge Agent
    knowledge_agent = KnowledgeAgent("knowledge_agent_001", {"knowledge_dir": "./workspace/knowledge"})
    knowledge_config = AgentConfig(
        agent_id="knowledge_agent_001",
        capabilities=["memory_management", "context_persistence"],
        max_concurrent_tasks=5,
    )
    await engine.register_agent(knowledge_config, knowledge_agent)
    agents["knowledge_agent_001"] = knowledge_agent
    logger.info("✅ Knowledge Agent registered")
    
    logger.info(f"🚀 All {len(agents)} agents initialized successfully")
    return agents
