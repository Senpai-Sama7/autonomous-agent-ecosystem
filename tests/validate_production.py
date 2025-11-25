"""
Production Validation Script
Verifies all components are operational and production-ready.
"""
import sys
import os
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    try:
        # Core modules
        from core.engine import AgentEngine, AgentConfig, Workflow, Task
        from core.database import DatabaseManager
        from core.nl_interface import NaturalLanguageInterface
        
        # Agents
        from agents.base_agent import BaseAgent, AgentCapability, TaskResult
        from agents.research_agent import ResearchAgent
        from agents.code_agent import CodeAgent
        from agents.filesystem_agent import FileSystemAgent
        
        # Monitoring
        from monitoring.monitoring_dashboard import MonitoringDashboard
        
        print("✅ All core imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_database():
    """Test database initialization"""
    print("\nTesting database...")
    try:
        from core.database import DatabaseManager
        db = DatabaseManager("test_ecosystem.db")
        db.save_agent("test_agent", {"test": "data"}, "active", 0.95)
        print("✅ Database operational")
        
        # Cleanup
        if os.path.exists("test_ecosystem.db"):
            os.remove("test_ecosystem.db")
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_filesystem_agent():
    """Test FileSystem agent"""
    print("\nTesting FileSystem Agent...")
    try:
        from agents.filesystem_agent import FileSystemAgent
        import asyncio
        
        async def test():
            agent = FileSystemAgent("test_fs", {"root_dir": "./test_workspace"})
            result = agent._write_file({"path": "test.txt", "content": "Hello, World!"})
            
            # Cleanup
            import shutil
            if os.path.exists("./test_workspace"):
                shutil.rmtree("./test_workspace")
                
            return result.success
            
        success = asyncio.run(test())
        if success:
            print("✅ FileSystem Agent operational")
        else:
            print("❌ FileSystem Agent failed")
        return success
    except Exception as e:
        print(f"❌ FileSystem Agent test failed: {e}")
        return False

def check_dependencies():
    """Check all required dependencies are installed"""
    print("\nChecking dependencies...")
    required = [
        'openai', 'duckduckgo_search', 'bs4', 'requests', 
        'sqlalchemy', 'dotenv', 'matplotlib', 'numpy',
        'customtkinter', 'PIL', 'yaml'
    ]
    
    missing = []
    for module in required:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies installed")
        return True

def test_engine_initialization():
    """Test engine can be initialized"""
    print("\nTesting Engine initialization...")
    try:
        from core.engine import AgentEngine
        engine = AgentEngine()
        print(f"✅ Engine initialized with {len(engine.agents)} agents")
        return True
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        return False

def verify_config_files():
    """Verify configuration files exist and are valid"""
    print("\nVerifying configuration files...")
    try:
        import yaml
        
        # Check system_config.yaml
        with open('../config/system_config.yaml', 'r') as f:
            sys_config = yaml.safe_load(f)
        
        # Check agents.yaml
        with open('../config/agents.yaml', 'r') as f:
            agents_config = yaml.safe_load(f)
            
        print(f"✅ Config files valid. Found {len(agents_config)} agent configs")
        return True
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        return False

def main():
    print("=" * 60)
    print("PRODUCTION VALIDATION - Autonomous Agent Ecosystem")
    print("=" * 60)
    
    tests = [
        ("Dependency Check", check_dependencies),
        ("Import Test", test_imports),
        ("Database Test", test_database),
        ("Engine Initialization", test_engine_initialization),
        ("FileSystem Agent", test_filesystem_agent),
        ("Config Files", verify_config_files),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("=" * 60)
    print(f"Result: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("✅ System is PRODUCTION READY")
        return 0
    else:
        print("❌ System has issues that need resolution")
        return 1

if __name__ == "__main__":
    sys.exit(main())
