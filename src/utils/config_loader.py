"""
Configuration Management for Autonomous Agent Ecosystem
"""
import os
import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("ConfigLoader")

@dataclass
class SystemConfig:
    """Global system configuration"""
    environment: str = "production"
    log_level: str = "INFO"
    database_path: str = "ecosystem.db"
    max_concurrent_workflows: int = 10
    enable_monitoring: bool = True
    
@dataclass
class LLMConfig:
    """LLM Provider Configuration"""
    provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3

class ConfigLoader:
    """Load and manage configuration from files and environment variables"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.system_config = SystemConfig()
        self.llm_config = LLMConfig()
        self.agent_configs: Dict[str, Any] = {}
        
    def load_configs(self) -> Dict[str, Any]:
        """Load all configurations"""
        self._load_system_config()
        self._load_agent_configs()
        return {
            "system": self.system_config,
            "llm": self.llm_config,
            "agents": self.agent_configs
        }

    def _load_system_config(self):
        """Load system-wide settings"""
        config_path = os.path.join(self.config_dir, "system_config.yaml")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if data:
                        # Update system config
                        sys_data = data.get('system', {})
                        self.system_config = SystemConfig(**{
                            k: v for k, v in sys_data.items() 
                            if k in self.system_config.__dict__
                        })
                        
                        # Update LLM config
                        llm_data = data.get('llm', {})
                        self.llm_config = LLMConfig(**{
                            k: v for k, v in llm_data.items() 
                            if k in self.llm_config.__dict__
                        })
            except Exception as e:
                logger.error(f"Failed to load system config: {e}")

        # Override with Environment Variables
        self.llm_config.api_key = os.getenv("OPENAI_API_KEY") or self.llm_config.api_key
        if os.getenv("LLM_PROVIDER"):
            self.llm_config.provider = os.getenv("LLM_PROVIDER")

    def _load_agent_configs(self):
        """Load individual agent configurations"""
        agents_path = os.path.join(self.config_dir, "agents.yaml")
        if os.path.exists(agents_path):
            try:
                with open(agents_path, 'r') as f:
                    self.agent_configs = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Failed to load agent configs: {e}")
