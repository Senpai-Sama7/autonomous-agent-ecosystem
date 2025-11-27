# ASTRO Examples

This directory contains runnable examples demonstrating ASTRO features.

## Quick Start

### Prerequisites

```bash
cd /Downloads/Astro-main
source venv/bin/activate

# Verify system is ready
./venv/bin/python health_check.py
```

### Example 1: Health Check

Verify all systems are operational:

```bash
./venv/bin/python health_check.py
```

Expected output includes:
- Network connectivity check
- Database integrity check
- Configuration validation
- Test suite execution
- GUI import verification
- Ollama LLM server check
- End-to-end LLM test
- Web search test

### Example 2: Run Integration Tests

Execute the advanced systems test suite:

```bash
./venv/bin/python -m pytest tests/test_advanced_systems.py -v
```

Expected: 7/7 tests pass

### Example 3: GUI Application

Launch the desktop GUI:

```bash
./venv/bin/python src/gui_app.py
```

Features to try:
1. Click "Start" to initialize agents
2. Type a request: "Search for Python tutorials"
3. View results in the Reasoning tab
4. Check detailed logs in the Logs tab

### Example 4: CLI Interactive Mode

Run in command-line interactive mode:

```bash
./venv/bin/python src/main.py --interactive --llm-provider ollama --model-name llama3.2
```

Example prompts:
- "Research the latest AI news"
- "Write a Python script to calculate fibonacci numbers"
- "Create a file called notes.md with a summary"

### Example 5: ResearchAgent Direct Test

Test the research agent directly:

```python
#!/usr/bin/env python3
import sys
import asyncio
sys.path.insert(0, 'src')

from agents.research_agent import ResearchAgent

async def main():
    agent = ResearchAgent('test', {'max_search_results': 3})
    task = {'payload': {'query': 'Python best practices 2025'}}
    result = await agent.execute_task(task, None)
    
    print(f"Success: {result.success}")
    if result.success:
        print(f"Sources: {len(result.result_data['sources'])}")
        print(f"Summary: {result.result_data['summary'][:200]}...")

asyncio.run(main())
```

Run with:
```bash
./venv/bin/python examples/research_agent_test.py
```

### Example 6: CodeAgent Direct Test

Test code generation:

```python
#!/usr/bin/env python3
import sys
import asyncio
sys.path.insert(0, 'src')

from agents.code_agent import CodeAgent

async def main():
    agent = CodeAgent('test', {
        'llm_provider': 'ollama',
        'model_name': 'llama3.2',
        'safe_mode': True
    })
    
    task = {
        'payload': {
            'code_task_type': 'generate_code',
            'requirements': 'Write a function that calculates factorial'
        }
    }
    
    result = await agent.execute_task(task, None)
    print(f"Success: {result.success}")
    if result.success:
        print(f"Generated code:\n{result.result_data['code']}")

asyncio.run(main())
```

### Example 7: LLM Factory

Test LLM client initialization:

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from core.llm_factory import LLMFactory

# Test Ollama client
client = LLMFactory.create_client('ollama')
print(f"Client type: {type(client).__name__}")
print(f"Module: {type(client).__module__}")

# Test sync client
sync_client = LLMFactory.create_sync_client('ollama')
print(f"Sync client type: {type(sync_client).__name__}")
```

### Example 8: Database Inspection

Inspect the ecosystem database:

```bash
# List tables
sqlite3 ecosystem.db ".tables"

# Show schema
sqlite3 ecosystem.db ".schema"

# Query agents
sqlite3 ecosystem.db "SELECT * FROM agents LIMIT 5"

# Query recent tasks
sqlite3 ecosystem.db "SELECT * FROM tasks ORDER BY id DESC LIMIT 10"
```

## Troubleshooting

### Ollama Not Running

```bash
# Start Ollama
ollama serve &

# Pull a model if needed
ollama pull llama3.2

# Verify
curl http://localhost:11434/api/tags
```

### Missing Dependencies

```bash
pip install -r requirements.txt
```

### GUI Display Issues

Ensure DISPLAY environment variable is set:

```bash
echo $DISPLAY  # Should show :0 or :1
```

## Performance Benchmarks

| Operation | Expected Time |
|-----------|---------------|
| Health check (full) | < 10s |
| Integration tests | < 2s |
| GUI startup | < 3s |
| LLM response (Ollama, 3B model) | < 2s |
| Web search | < 3s |
