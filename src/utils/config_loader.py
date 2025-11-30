"""
Configuration Management for Autonomous Agent Ecosystem
Updated by C0Di3 to support Declarative Workflows and Templating.
"""
import os
import yaml
import json
import uuid
import time
import copy
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

# Import Core Definitions for Factory Generation
from core.engine import Workflow, Task, WorkflowPriority

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
        self.workflow_templates: Dict[str, Any] = {}
        
    def load_configs(self) -> Dict[str, Any]:
        """Load all configurations including system, agents, and workflows"""
        self._load_system_config()
        self._load_agent_configs()
        self._load_workflow_templates()
        
        return {
            "system": self.system_config,
            "llm": self.llm_config,
            "agents": self.agent_configs,
            "workflows": self.workflow_templates
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

    def _load_workflow_templates(self):
        """Load declarative workflow templates from YAML"""
        workflows_path = os.path.join(self.config_dir, "workflows.yaml")
        if os.path.exists(workflows_path):
            try:
                with open(workflows_path, 'r') as f:
                    data = yaml.safe_load(f) or {}
                    self.workflow_templates = data.get("workflows", {})
                logger.info(f"Loaded {len(self.workflow_templates)} workflow templates")
            except Exception as e:
                logger.error(f"Failed to load workflows.yaml: {e}")
        else:
            logger.warning(f"No workflows.yaml found at {workflows_path}")

    def load_agent_configs(self) -> Dict[str, Any]:
        """Public method to load and return agent configurations"""
        self._load_agent_configs()
        return self.agent_configs

    # --- Factory Method for Agentic Workflow Generation ---

    def create_workflow_from_template(self, template_name: str, variables: Dict[str, str] = None) -> Optional[Workflow]:
        """
        Instantiates a Workflow object from a YAML template, substituting variables.
        
        Args:
            template_name: The key in workflows.yaml (e.g., 'research_topic')
            variables: Dict of values to replace {{ key }} in the template
            
        Returns:
            A Workflow object ready for AgentEngine.submit_workflow()
        """
        if template_name not in self.workflow_templates:
            logger.error(f"Workflow template '{template_name}' not found.")
            return None

        template = copy.deepcopy(self.workflow_templates[template_name])
        variables = variables or {}
        
        # 1. Generate Unique ID for this execution instance
        run_id = str(uuid.uuid4())[:8]
        workflow_id = f"wf_{template_name}_{run_id}"
        
        # 2. Parse Priority
        priority_str = template.get("priority", "medium").upper()
        try:
            priority = WorkflowPriority[priority_str]
        except KeyError:
            priority = WorkflowPriority.MEDIUM

        # 3. Build Tasks
        tasks = []
        raw_tasks = template.get("tasks", [])
        
        for i, raw_task in enumerate(raw_tasks):
            # Resolve task ID (must be unique per run)
            task_local_id = raw_task.get("id", f"task_{i}")
            task_global_id = f"{workflow_id}_{task_local_id}"
            
            # Perform Variable Substitution in instruction/payload
            instruction = raw_task.get("instruction", "")
            payload_template = raw_task.get("payload", {})
            
            # Simple Jinja2-style replacement
            for var_key, var_val in variables.items():
                placeholder = f"{{{{ {var_key} }}}}"
                if placeholder in instruction:
                    instruction = instruction.replace(placeholder, str(var_val))
                
                # Also substitute in payload string values
                for pk, pv in payload_template.items():
                    if isinstance(pv, str) and placeholder in pv:
                        payload_template[pk] = pv.replace(placeholder, str(var_val))

            # Map YAML 'capability' string to list required by Engine
            cap = raw_task.get("capability")
            capabilities = [cap] if cap else []
            
            # Resolve dependencies (map local IDs to global IDs)
            deps = raw_task.get("dependencies", [])
            global_deps = [f"{workflow_id}_{d}" for d in deps]

            tasks.append(Task(
                task_id=task_global_id,
                description=instruction,
                required_capabilities=capabilities,
                priority=priority,
                dependencies=global_deps,
                payload=payload_template,
                workflow_id=workflow_id
            ))

        # 4. Construct Workflow Object
        return Workflow(
            workflow_id=workflow_id,
            name=f"{template_name} ({variables.get('query', 'Custom')})",
            tasks=tasks,
            priority=priority,
            cost_budget=template.get("budget", 100.0)
        )
