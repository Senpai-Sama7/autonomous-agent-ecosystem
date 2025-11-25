# Autonomous Agent Ecosystem

A production-ready, fully autonomous multi-agent system with natural language control, real web search, code execution, and persistent storage.

## 🚀 Production Features

### ✅ Real Capabilities (No Simulations)

- **Web Search & Scraping**: DuckDuckGo API + BeautifulSoup for live internet research
- **Code Generation**: OpenAI/OpenRouter/Ollama integration for generating Python code
- **Code Execution**: Safe subprocess execution with security sandboxing
- **File Operations**: Secure filesystem operations with path validation
- **Persistent Storage**: SQLite database with indexed tables
- **Natural Language Control**: LLM-powered intent parsing for plain English commands
- **Modern GUI**: Clay-morphic design with real-time monitoring

### 🎯 Agent Types

1. **Research Agent**: Web search, content extraction, knowledge synthesis
2. **Code Agent**: Code generation, optimization, debugging, execution
3. **FileSystem Agent**: Safe file read/write/list operations

## 📦 Installation

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

Create a `.env` file:

```env
OPENAI_API_KEY=your_key_here
OPENROUTER_API_KEY=your_openrouter_key
```

### 3. Validate Installation

```powershell
python tests/validate_production.py
```

## 🎮 Usage

### CLI Mode (Default)

```powershell
python src/main.py --duration 120
```

### Interactive Natural Language Mode

```powershell
# With OpenAI
python src/main.py --interactive --api-key "sk-..."

# With Ollama (Local)
python src/main.py --interactive --llm-provider ollama --model-name llama3

# With OpenRouter
python src/main.py --interactive --llm-provider openrouter --api-key "..." --model-name "anthropic/claude-3-opus"
```

### GUI Mode

```powershell
python src/gui_app.py
```

## 🛠️ Configuration

### System Config (`config/system_config.yaml`)

```yaml
system:
  environment: "production"
  log_level: "INFO"
  max_concurrent_workflows: 10

llm:
  provider: "openai"
  model_name: "gpt-3.5-turbo"
  timeout: 60
```

### Agent Config (`config/agents.yaml`)

```yaml
research_agent_001:
  max_search_results: 6
  max_pages_to_scrape: 4
  quality_threshold: 0.7
```

## 📊 Architecture

```
┌─────────────────────────────────────┐
│   Natural Language Interface        │
│   (LLM-powered intent parsing)      │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│      Agent Engine (Core)            │
│  - Task Queue & Scheduling          │
│  - Dependency Tracking              │
│  - Failure Recovery                 │
│  - Incentive System                 │
└─────────────────┬───────────────────┘
                  │
        ┌─────────┴─────────┬─────────────┐
        │                   │             │
    ┌───▼────┐       ┌──────▼───┐   ┌────▼─────┐
    │Research│       │   Code   │   │FileSystem│
    │ Agent  │       │  Agent   │   │  Agent   │
    └────────┘       └──────────┘   └──────────┘
```

## 🔧 Production Deployment

### Database

- **Default**: SQLite (`ecosystem.db`)
- **Production**: PostgreSQL recommended
- **Indices**: Automatically created on workflow_id, agent_id, timestamps

### Security

- **Code Execution**: Safe mode enabled by default (blocks `os`, `sys`, `subprocess` imports)
- **Filesystem**: Sandboxed to `./workspace` directory
- **API Keys**: Loaded from environment variables

### Monitoring

- Real-time dashboard tracking agent health, task completion, system metrics
- Persistent metrics in database
- Configurable alerting thresholds

## 📝 Example Workflows

### Research & Code Generation

```python
"Research the latest quantum computing algorithms and generate a Python implementation"
```

This automatically:

1. Searches the web for quantum computing info
2. Generates Python code based on findings
3. Optionally executes and tests the code

### File Operations

```python
"Create a report summarizing agent performance and save it to reports/summary.md"
```

## 🧪 Testing

Run the validation suite:

```powershell
python tests/validate_production.py
```

Expected output:

```
✅ PASS - Dependency Check
✅ PASS - Import Test
✅ PASS - Database Test
✅ PASS - Engine Initialization
✅ PASS - FileSystem Agent
✅ PASS - Config Files

Result: 6/6 tests passed (100%)
✅ System is PRODUCTION READY
```

## 📂 Project Structure

```
agent_ecosystem/
├── config/
│   ├── system_config.yaml
│   └── agents.yaml
├── src/
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── research_agent.py
│   │   ├── code_agent.py
│   │   └── filesystem_agent.py
│   ├── core/
│   │   ├── engine.py
│   │   ├── database.py
│   │   └── nl_interface.py
│   ├── monitoring/
│   │   └── monitoring_dashboard.py
│   ├── utils/
│   │   ├── config_loader.py
│   │   ├── logger.py
│   │   └── helpers.py
│   ├── main.py
│   └── gui_app.py
├── tests/
│   └── validate_production.py
├── requirements.txt
└── README.md
```

## 🔒 Security Considerations

1. **Code Execution**: Uses subprocess with 10s timeout and import restrictions
2. **File Access**: Limited to workspace directory
3. **API Keys**: Never hardcoded, always from environment
4. **Database**: SQL injection protected via parameterized queries

## 🚨 Known Limitations

- Web scraping may fail on JavaScript-heavy sites
- Code execution sandbox is subprocess-based (not container-level isolation)
- SQLite not suitable for high-concurrency production (use PostgreSQL)

## 🔮 Future Enhancements

- Docker containerization for code execution
- PostgreSQL migration
- Web API (FastAPI)
- Kubernetes deployment manifests
- Advanced ML-based anomaly detection

## 📄 License

MIT License - See LICENSE file for details

---

**Status**: ✅ Production Ready (v1.0.0)
