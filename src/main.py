"""
Autonomous Agent Ecosystem - Main Application
November 25, 2025

This is the main entry point for the autonomous agent ecosystem system.
It demonstrates the complete end-to-end functionality of the system
with real agents, workflows, and monitoring capabilities.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List
import argparse
import os

from core.engine import AgentEngine, AgentConfig, Workflow, Task, WorkflowPriority
from agents.research_agent import ResearchAgent
from agents.code_agent import CodeAgent
from agents.filesystem_agent import FileSystemAgent
from monitoring.monitoring_dashboard import MonitoringDashboard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MainApplication")

class AutonomousAgentEcosystem:
    """
    Main application class that orchestrates the entire autonomous agent ecosystem.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the ecosystem"""
        self.engine = AgentEngine()
        self.dashboard = MonitoringDashboard()
        self.agents = {}
        self.workflows = {}
        self.config = config or {}
        
        logger.info("Autonomous Agent Ecosystem initialized")
        
    async def initialize_agents(self):
        """Initialize and register all agents"""
        logger.info("Initializing agents...")
        
        # Load agent configs from YAML
        from utils.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        agent_configs = config_loader.load_agent_configs()
        
        # Research Agent
        research_config_dict = agent_configs.get('research_agent_001', {})
        research_agent = ResearchAgent(
            agent_id="research_agent_001",
            config=research_config_dict
        )
        
        # Code Agent Configuration
        code_config_dict = agent_configs.get('code_agent_001', {})
        # Merge with LLM settings from command line
        code_config_dict.update({
            'llm_provider': self.config.get('llm_provider', 'openai'),
            'model_name': self.config.get('model_name'),
            'api_key': self.config.get('api_key'),
            'api_base': self.config.get('api_base')
        })
        
        code_agent = CodeAgent(
            agent_id="code_agent_001",
            config=code_config_dict
        )

        # FileSystem Agent
        fs_config_dict = agent_configs.get('filesystem_agent_001', {})
        fs_agent = FileSystemAgent(
            agent_id="filesystem_agent_001",
            config=fs_config_dict
        )
        
        # Register agents with the engine
        research_engine_config = AgentConfig(
            agent_id="research_agent_001",
            capabilities=["web_search", "content_extraction", "knowledge_synthesis", "data_processing"],
            max_concurrent_tasks=research_config_dict.get('max_concurrent_tasks', 2),
            reliability_score=0.92,
            cost_per_operation=1.5
        )
        
        code_engine_config = AgentConfig(
            agent_id="code_agent_001",
            capabilities=["code_generation", "code_optimization", "debugging", "optimization"],
            max_concurrent_tasks=code_config_dict.get('max_concurrent_tasks', 2),
            reliability_score=0.88,
            cost_per_operation=2.0
        )

        fs_engine_config = AgentConfig(
            agent_id="filesystem_agent_001",
            capabilities=["data_processing", "file_operations"],
            max_concurrent_tasks=fs_config_dict.get('max_concurrent_tasks', 5),
            reliability_score=0.99,
            cost_per_operation=0.5
        )
        
        await self.engine.register_agent(research_engine_config, instance=research_agent)
        await self.engine.register_agent(code_engine_config, instance=code_agent)
        await self.engine.register_agent(fs_engine_config, instance=fs_agent)
        
        # Store agent references
        self.agents["research_agent_001"] = research_agent
        self.agents["code_agent_001"] = code_agent
        self.agents["filesystem_agent_001"] = fs_agent
        
        # Register agents with monitoring dashboard
        self.dashboard.register_agent("research_agent_001", {
            'state': 'active',
            'performance_score': 0.92,
            'capabilities': ['web_search', 'content_extraction']
        })
        
        self.dashboard.register_agent("code_agent_001", {
            'state': 'active',
            'performance_score': 0.88,
            'capabilities': ['code_generation', 'optimization']
        })

        self.dashboard.register_agent("filesystem_agent_001", {
            'state': 'active',
            'performance_score': 0.99,
            'capabilities': ['file_operations']
        })
        
        logger.info("Agents initialized and registered successfully")
        
    async def create_sample_workflows(self):
        """Create sample workflows to demonstrate system capabilities"""
        logger.info("Creating sample workflows...")
        
        # Workflow 1: Research and Code Generation for ML Task
        ml_workflow = Workflow(
            workflow_id="workflow_ml_001",
            name="ML Model Research and Implementation",
            tasks=[
                Task(
                    task_id="task_research_001",
                    description="Research latest ML model architectures for time series forecasting",
                    required_capabilities=["data_processing"], # Matches ResearchAgent capability
                    priority=WorkflowPriority.HIGH,
                    deadline=time.time() + 300,  # 5 minutes
                    payload={
                        'query': 'state of the art time series forecasting models 2025',
                    }
                ),
                Task(
                    task_id="task_code_001",
                    description="Generate Python code for LSTM time series forecasting model",
                    required_capabilities=["optimization"], # Matches CodeAgent capability
                    priority=WorkflowPriority.HIGH,
                    deadline=time.time() + 600,  # 10 minutes
                    dependencies=["task_research_001"],
                    payload={
                        'code_task_type': 'generate_code',
                        'requirements': 'Create a simple python script that prints "Hello from Autonomous Agent Ecosystem!"'
                    }
                ),
                Task(
                    task_id="task_exec_001",
                    description="Execute the generated code",
                    required_capabilities=["optimization"],
                    priority=WorkflowPriority.HIGH,
                    dependencies=["task_code_001"],
                    payload={
                        'code_task_type': 'execute_code',
                        'code': 'print("Hello from the executed code!")' # In real flow, this would come from previous task result
                    }
                )
            ],
            priority=WorkflowPriority.HIGH,
            max_execution_time=900,  # 15 minutes
            cost_budget=500.0,
            success_criteria={
                'min_quality_score': 0.8,
                'max_execution_time': 600,
                'required_capabilities': ['time_series_forecasting', 'model_evaluation']
            }
        )
        
        # Submit workflows to engine
        await self.engine.submit_workflow(ml_workflow)
        
        self.workflows["workflow_ml_001"] = ml_workflow
        
        logger.info("Sample workflows created and submitted successfully")
        
    async def start_system(self):
        """Start the entire autonomous agent ecosystem"""
        logger.info("🚀 Starting Autonomous Agent Ecosystem (PRODUCTION MODE)")
        logger.info("=" * 60)
        
        try:
            # Initialize agents
            await self.initialize_agents()
            
            # Create sample workflows
            await self.create_sample_workflows()
            
            # Start engine and monitoring in parallel
            engine_task = asyncio.create_task(self.engine.start_engine())
            monitoring_task = asyncio.create_task(self.dashboard.start_monitoring())
            
            # Run for a specific duration or until interrupted
            logger.info("System is now running. Press Ctrl+C to stop...")
            logger.info("=" * 60)
            
            # Run for specified duration
            duration = self.config.get('duration', 120)
            await asyncio.sleep(duration)
            
            # Cancel tasks gracefully
            engine_task.cancel()
            monitoring_task.cancel()
            
            try:
                await engine_task
                await monitoring_task
            except asyncio.CancelledError:
                pass
                
            logger.info("✅ System shutdown completed gracefully")
            
        except KeyboardInterrupt:
            logger.info("🛑 System interrupted by user")
        except Exception as e:
            logger.error(f"❌ System error: {str(e)}", exc_info=True)
        finally:
            await self.generate_final_report()
            
    async def generate_final_report(self):
        """Generate a comprehensive final report of system execution"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 FINAL SYSTEM REPORT")
        logger.info("=" * 60)
        
        # System health summary
        system_health = self.dashboard.get_system_health()
        logger.info(f"📈 System Health Score: {system_health['health_score']:.2f} ({system_health['status']})")
        logger.info(f"🤖 Active Agents: {system_health['active_agents_count']}")
        logger.info(f"⚠️  Recent Alerts: {len(system_health['recent_alerts'])}")
        
        # Agent performance summary
        logger.info("\n" + "-" * 40)
        logger.info("_AGENT PERFORMANCE SUMMARY_")
        logger.info("-" * 40)
        
        for agent_id in self.agents.keys():
            performance = self.dashboard.get_agent_performance(agent_id)
            logger.info(f"\n{agent_id}:")
            logger.info(f"  Current Performance: {performance.get('current_performance', 0):.2f}")
            logger.info(f"  Average CPU Usage: {performance.get('avg_cpu_usage', 0):.2f}")
            logger.info(f"  Average Memory: {performance.get('avg_memory_usage', 0):.2f}")
            logger.info(f"  Status: {performance.get('status', 'unknown')}")
            logger.info(f"  Recent Alerts: {performance.get('recent_alerts', 0)}")
        
        # Workflow execution summary
        logger.info("\n" + "-" * 40)
        logger.info("_WORKFLOW EXECUTION SUMMARY_")
        logger.info("-" * 40)
        
        for workflow_id, workflow in self.workflows.items():
            logger.info(f"\n{workflow.name} ({workflow_id}):")
            logger.info(f"  Priority: {workflow.priority.value}")
            logger.info(f"  Tasks: {len(workflow.tasks)}")
            logger.info(f"  Budget: ${workflow.cost_budget:.2f}")
            logger.info(f"  Max Time: {workflow.max_execution_time:.0f}s")
        
        # Alert summary
        alert_summary = self.dashboard.get_alert_summary()
        logger.info("\n" + "-" * 40)
        logger.info("_ALERT SUMMARY_")
        logger.info("-" * 40)
        logger.info(f"Total Alerts: {alert_summary['total_alerts']}")
        logger.info(f"Health Impact: {alert_summary['health_impact']:.2f}")
        logger.info("Severity Distribution:")
        for severity, count in alert_summary['severity_counts'].items():
            if count > 0:
                logger.info(f"  {severity.upper()}: {count}")
        
        # System metrics
        logger.info("\n" + "-" * 40)
        logger.info("_SYSTEM METRICS_")
        logger.info("-" * 40)
        metrics = self.engine.get_system_metrics()
        logger.info(f"Total Agents: {metrics['agent_count']}")
        logger.info(f"Active Workflows: {metrics['active_workflows']}")
        logger.info(f"Pending Tasks: {metrics['pending_tasks']}")
        logger.info(f"Active Tasks: {metrics['active_tasks']}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✨ AUTONOMOUS AGENT ECOSYSTEM EXECUTION COMPLETE")
        logger.info("=" * 60)

from core.llm_factory import LLMFactory

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Autonomous Agent Ecosystem')
    parser.add_argument('--duration', type=int, default=120,
                      help='Duration to run the system in seconds (default: 120)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    parser.add_argument('--test-mode', action='store_true',
                      help='Run in test mode with simulated data')
    parser.add_argument('--interactive', action='store_true',
                      help='Run in interactive mode to accept natural language commands')
    
    # LLM Configuration
    parser.add_argument('--llm-provider', type=str, default='openai', choices=['openai', 'openrouter', 'ollama'],
                      help='LLM provider to use (openai, openrouter, ollama)')
    parser.add_argument('--model-name', type=str, default='gpt-3.5-turbo',
                      help='Model name to use (e.g., gpt-4, anthropic/claude-3-opus, llama3)')
    parser.add_argument('--api-key', type=str,
                      help='API key for the LLM provider')
    parser.add_argument('--api-base', type=str,
                      help='Base URL for the LLM API (useful for Ollama or custom endpoints)')
    
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_arguments()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert args to config dict
    config = vars(args)
    
    ecosystem = AutonomousAgentEcosystem(config=config)
    
    try:
        # Initialize agents
        await ecosystem.initialize_agents()
        
        # Start engine and monitoring
        engine_task = asyncio.create_task(ecosystem.engine.start_engine())
        monitoring_task = asyncio.create_task(ecosystem.dashboard.start_monitoring())
        
        if args.interactive:
            # Initialize NL Interface using Factory
            llm_client = LLMFactory.create_client(
                provider=args.llm_provider,
                api_key=args.api_key,
                base_url=args.api_base
            )

            nl_interface = NaturalLanguageInterface(
                engine=ecosystem.engine, 
                llm_client=llm_client,
                model_name=args.model_name
            )
            
            logger.info("\n💬 Interactive Mode Enabled. Type your request (or 'exit' to quit):")
            while True:
                user_input = await asyncio.get_event_loop().run_in_executor(None, input, ">> ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                
                try:
                    workflow_id = await nl_interface.process_request(user_input)
                    print(f"✅ Workflow submitted: {workflow_id}")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
        else:
            # Create sample workflows and run for duration
            await ecosystem.create_sample_workflows()
            logger.info(f"System running for {args.duration} seconds...")
            await asyncio.sleep(args.duration)
            
        # Cancel tasks gracefully
        engine_task.cancel()
        monitoring_task.cancel()
        
        try:
            await engine_task
            await monitoring_task
        except asyncio.CancelledError:
            pass
            
        logger.info("✅ System shutdown completed gracefully")
        
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
    finally:
        await ecosystem.generate_final_report()

if __name__ == "__main__":
    """Run the main application"""
    print("""
    🚀 AUTONOMOUS AGENT ECOSYSTEM - PRODUCTION MODE 🚀
    
    Features Enabled:
    ✅ Real Web Search (DuckDuckGo)
    ✅ Content Extraction (BeautifulSoup)
    ✅ Real Code Generation (OpenAI, OpenRouter, Ollama)
    ✅ Safe Code Execution (Subprocess)
    ✅ Persistent Storage (SQLite)
    ✅ Natural Language Interface (Interactive Mode)
    
    The system is now starting...
    """)
    
    asyncio.run(main())