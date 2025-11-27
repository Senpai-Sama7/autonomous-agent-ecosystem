"""ASTRO Health Check

Comprehensive production readiness verification:
- Network connectivity
- Ollama LLM server status
- Database integrity
- Config file validation
- Advanced systems tests (pytest)
- GUI module imports
- End-to-end LLM test (optional)
"""
import os
import sys
import subprocess
import sqlite3

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


def check_network() -> bool:
    """Verify network connectivity."""
    print("== Checking network connectivity ==")
    try:
        import requests
        resp = requests.get("https://duckduckgo.com", timeout=5)
        if resp.status_code == 200:
            print("  ✓ Network connectivity OK")
            return True
        print(f"  ✗ Network check failed: HTTP {resp.status_code}")
        return False
    except Exception as e:
        print(f"  ✗ Network check failed: {e}")
        return False


def check_ollama() -> tuple[bool, list[str]]:
    """Check if Ollama is running and list available models."""
    print("\n== Checking Ollama LLM server ==")
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            print(f"  ✓ Ollama running with {len(models)} model(s)")
            for m in models[:5]:
                print(f"    - {m}")
            if len(models) > 5:
                print(f"    ... and {len(models) - 5} more")
            return True, models
        print(f"  ✗ Ollama responded with HTTP {resp.status_code}")
        return False, []
    except Exception as e:
        print(f"  ⚠ Ollama not running: {e}")
        return False, []


def check_database() -> bool:
    """Verify SQLite database integrity."""
    print("\n== Checking database ==")
    db_path = os.path.join(REPO_ROOT, "ecosystem.db")
    if not os.path.exists(db_path):
        print(f"  ✗ Database not found: {db_path}")
        return False
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        required = {"agents", "workflows", "tasks", "metrics"}
        missing = required - set(tables)
        if missing:
            print(f"  ✗ Missing tables: {missing}")
            return False
        print(f"  ✓ Database OK ({len(tables)} tables: {', '.join(sorted(tables))})")
        return True
    except Exception as e:
        print(f"  ✗ Database error: {e}")
        return False


def check_configs() -> bool:
    """Validate YAML configuration files."""
    print("\n== Checking configuration files ==")
    try:
        import yaml
        configs = ["config/system_config.yaml", "config/agents.yaml"]
        for cfg in configs:
            path = os.path.join(REPO_ROOT, cfg)
            with open(path) as f:
                yaml.safe_load(f)
            print(f"  ✓ {cfg} valid")
        return True
    except Exception as e:
        print(f"  ✗ Config error: {e}")
        return False


def run_pytest() -> bool:
    """Run the advanced systems pytest suite."""
    print("\n== Running advanced systems tests ==")
    cmd = [sys.executable, "-m", "pytest", "tests/test_advanced_systems.py", "-q"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode == 0:
        # Extract pass count from output
        lines = result.stdout.strip().split('\n')
        summary = [l for l in lines if 'passed' in l]
        print(f"  ✓ {summary[-1] if summary else 'All tests passed'}")
        return True
    else:
        print(f"  ✗ Tests FAILED")
        print(result.stdout)
        return False


def check_gui_imports() -> bool:
    """Verify that core GUI modules can be imported."""
    print("\n== Checking GUI imports ==")
    try:
        from gui_app import (
            EnhancedGUI,
            SettingsDialog,
            HelpDialog,
            TutorialOverlay,
            ErrorPopup,
            TUTORIAL_STEPS,
            COLORS,
        )
        print(f"  ✓ GUI modules OK ({len(TUTORIAL_STEPS)} tutorial steps, {len(COLORS)} colors)")
        return True
    except Exception as exc:
        print(f"  ✗ GUI import failed: {exc}")
        return False


def check_llm_e2e(models: list[str]) -> bool:
    """Run end-to-end LLM test with Ollama."""
    print("\n== Running end-to-end LLM test ==")
    if not models:
        print("  ⚠ Skipped (no Ollama models available)")
        return True  # Not a failure, just skipped
    
    try:
        import asyncio
        from core.llm_factory import LLMFactory
        
        client = LLMFactory.create_client("ollama")
        model = models[0].split(":")[0]  # Use first available model
        
        async def test_call():
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say 'OK' and nothing else."}],
                max_tokens=10
            )
            return response.choices[0].message.content
        
        result = asyncio.run(test_call())
        if result and len(result) > 0:
            print(f"  ✓ LLM responded: '{result.strip()[:50]}'")
            return True
        print(f"  ✗ LLM returned empty response")
        return False
    except Exception as e:
        print(f"  ✗ LLM test failed: {e}")
        return False


def check_web_search() -> bool:
    """Test web search capability."""
    print("\n== Testing web search ==")
    try:
        # Try new ddgs package first, fall back to duckduckgo_search
        try:
            from ddgs import DDGS
            results = DDGS().text("Python programming", max_results=2)
        except ImportError:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text("Python programming", max_results=2))
        
        if results:
            print(f"  ✓ Web search OK ({len(results)} results)")
            return True
        print("  ✗ Web search returned no results")
        return False
    except Exception as e:
        print(f"  ✗ Web search failed: {e}")
        return False


def main() -> int:
    print("=" * 50)
    print("ASTRO Comprehensive Health Check")
    print("=" * 50 + "\n")
    
    results = {}
    
    # Core checks (required)
    results["network"] = check_network()
    results["database"] = check_database()
    results["configs"] = check_configs()
    results["pytest"] = run_pytest()
    results["gui"] = check_gui_imports()
    
    # Optional checks (Ollama-dependent)
    ollama_ok, models = check_ollama()
    results["ollama"] = ollama_ok
    
    if ollama_ok:
        results["llm_e2e"] = check_llm_e2e(models)
    
    results["web_search"] = check_web_search()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    required = ["database", "configs", "pytest", "gui"]
    optional = ["network", "ollama", "llm_e2e", "web_search"]
    
    required_pass = all(results.get(k, False) for k in required)
    optional_pass = sum(1 for k in optional if results.get(k, False))
    
    for k in required:
        status = "✓" if results.get(k, False) else "✗"
        print(f"  [{status}] {k} (required)")
    
    for k in optional:
        if k in results:
            status = "✓" if results[k] else "⚠"
            print(f"  [{status}] {k} (optional)")
    
    print()
    if required_pass:
        print(f"✅ HEALTH CHECK PASSED - {optional_pass}/{len(optional)} optional checks OK")
        print("   System is production ready!")
        return 0
    else:
        print("❌ HEALTH CHECK FAILED - Required checks did not pass")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
