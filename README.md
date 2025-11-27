<div align="center">

# 🌟 ASTRO - Autonomous Agent Ecosystem

### *Your Personal AI Team That Actually Gets Things Done*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Senpai-Sama7/autonomous-agent-ecosystem/pulls)

<img src="https://img.shields.io/badge/AI%20Powered-OpenAI%20|%20Ollama%20|%20OpenRouter-purple" alt="AI Powered">

---

**Imagine having a team of AI assistants that can research the internet, write code, and manage your files — all from a single command in plain English.**

[Get Started](#-quick-start-5-minutes) • [Watch Demo](#-see-it-in-action) • [User Guide](#-complete-user-guide) • [FAQ](#-frequently-asked-questions)

</div>

---

## 🎬 See It In Action

```
You: "Research the latest trends in AI and create a summary report"

🔍 Research Agent: Searching the web...
📝 Research Agent: Found 6 relevant sources
💾 FileSystem Agent: Created reports/ai_trends_2025.md
✅ Done! Your report is ready.
```

**That's it.** No complex commands. No programming required. Just tell it what you want.

---

## ✨ What Can ASTRO Do For You?

<table>
<tr>
<td width="33%" align="center">

### 🔬 Research Agent
**Your Personal Researcher**

- Searches the entire internet
- Reads and summarizes articles
- Finds the information you need

*"Find me the best pizza recipe"*

</td>
<td width="33%" align="center">

### 💻 Code Agent
**Your AI Programmer**

- Writes Python code for you
- Runs and tests the code
- Fixes bugs automatically

*"Create a script to organize my photos"*

</td>
<td width="33%" align="center">

### 📁 File Agent
**Your Digital Secretary**

- Creates and edits files
- Organizes your documents
- Saves reports and summaries

*"Save this summary to my reports folder"*

</td>
</tr>
</table>

---

## 🚀 Quick Start (5 Minutes)

### What You'll Need

- ✅ A computer (Windows, Mac, or Linux)
- ✅ Python installed ([Download here](https://python.org/downloads) if you don't have it)
- ✅ An API key (free options available!)

### Step 1: Download ASTRO

**Option A: Download ZIP** (Easiest)
1. Click the green "Code" button above
2. Click "Download ZIP"
3. Extract the folder to your Desktop

**Option B: Use Git** (For developers)
```bash
git clone https://github.com/Senpai-Sama7/autonomous-agent-ecosystem.git
cd autonomous-agent-ecosystem
```

### Step 2: Install Requirements

Open your terminal (Command Prompt on Windows, Terminal on Mac) and run:

```bash
pip install -r requirements.txt
```

> 💡 **Tip**: If you see an error, try `pip3` instead of `pip`

### Step 3: Set Up Your AI Provider

Choose ONE of these options:

<details>
<summary><b>🆓 Option A: Use Ollama (FREE - Runs on Your Computer)</b></summary>

1. Download Ollama from [ollama.ai](https://ollama.ai)
2. Install it (just double-click the installer)
3. Open terminal and run: `ollama pull llama3`
4. That's it! No API key needed.

</details>

<details>
<summary><b>💳 Option B: Use OpenAI (Best Quality)</b></summary>

1. Go to [platform.openai.com](https://platform.openai.com)
2. Create an account and add billing
3. Go to API Keys and create a new key
4. Create a file named `.env` in the ASTRO folder with:
```
OPENAI_API_KEY=sk-your-key-here
```

</details>

<details>
<summary><b>🔄 Option C: Use OpenRouter (Many Models)</b></summary>

1. Go to [openrouter.ai](https://openrouter.ai)
2. Create an account
3. Get your API key
4. Create a file named `.env` in the ASTRO folder with:
```
OPENROUTER_API_KEY=your-key-here
```

</details>

### Step 4: Launch ASTRO!

**For the Beautiful GUI (Recommended):**
```bash
python src/gui_app.py
```

**For Command Line:**
```bash
python src/main.py --interactive
```

🎉 **Congratulations!** You're ready to use your personal AI team!

---

## 📖 Complete User Guide

### Using the Desktop App (GUI)

When you launch `gui_app.py`, you'll see a beautiful dark-themed interface:

```
┌─────────────────────────────────────────────────────────────┐
│  AGENT ECO                              [System Online 🟢]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  💬 Ask Your Agents                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Type your command here...                       [▶] │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  📡 System Activity          │  📜 Workflow History        │
│  ─────────────────────────── │  ────────────────────────   │
│  10:30 - Research started... │  🟡 Research AI trends      │
│  10:31 - Found 5 sources...  │  ✅ Generate report         │
│  10:32 - Task completed ✅   │  ⏳ File organization       │
│                              │                             │
└─────────────────────────────────────────────────────────────┘
```

**How to Use:**

1. **Click "🚀 Start System"** - Wait for "System Online" to appear
2. **Type your request** - Use plain English, like talking to a person
3. **Press Enter or click "Execute ▶"** - Watch the magic happen
4. **See results** - Check the Activity log and your files

### Example Commands You Can Try

| What You Want | What to Type |
|---------------|--------------|
| Search the web | *"Search for the best laptop under $1000"* |
| Create a file | *"Create a file called notes.txt with my meeting notes"* |
| Write code | *"Write a Python script that calculates my monthly budget"* |
| Research + Save | *"Research climate change and save a summary to climate_report.md"* |
| Complex tasks | *"Find Python tutorials, summarize the best ones, and save to learning_path.md"* |

### Understanding the Interface

| Section | What It Shows |
|---------|---------------|
| **System Status** (top-left) | 🟢 Green = Running, ⚫ Gray = Offline |
| **Agent Cards** (left sidebar) | Shows each AI agent and what it's doing |
| **Command Input** (top) | Where you type your requests |
| **Reasoning & Logs** (bottom-left) | AI chain-of-thought and detailed system logs |
| **Workflow History** (bottom-right) | List of your past commands |

### Settings & Configuration

Click **"⚙️ Settings"** to configure:

- **Provider**: Choose your AI (OpenAI, Ollama, or OpenRouter)
- **Model**: Select which AI model to use
- **API Key**: Enter your API key (hidden for security)

### Reasoning, Logs & Errors

- **Reasoning** tab shows the AI's Chain of Thought for each request.
- **Logs** tab shows detailed system and debug logs.
- When something goes wrong, an **error popup** appears with suggested fixes
  (for example: open Settings to set an API key, start Ollama, or switch models).

---

## 🎯 Tips for Best Results

### ✅ DO: Be Specific
```
Good: "Search for Python tutorials for beginners and save the top 5 to tutorials.md"
Bad:  "Find stuff"
```

### ✅ DO: Use Natural Language
```
Good: "Create a summary of today's tech news"
Good: "Help me write code to sort a list of names"
Bad:  "RUN_SEARCH --query='tech' --output=summary"
```

### ✅ DO: Chain Tasks Together
```
"Research machine learning, then create a Python example, and save everything to ml_notes.md"
```

### ❌ DON'T: Ask for Harmful Content
The system has built-in safety measures and won't:
- Access files outside the workspace folder
- Run dangerous code
- Perform harmful searches

---

## ❓ Frequently Asked Questions

<details>
<summary><b>Is my data safe?</b></summary>

Yes! ASTRO:
- Only accesses files in the `workspace` folder
- Never sends your files to the internet
- Keeps API keys encrypted locally
- Runs code in a secure sandbox

</details>

<details>
<summary><b>Do I need to pay?</b></summary>

Not necessarily! You have free options:
- **Ollama**: Completely free, runs on your computer
- **OpenAI**: Pay-per-use, usually pennies per request
- **OpenRouter**: Some free models available

</details>

<details>
<summary><b>Why isn't it working?</b></summary>

Try these fixes:

1. **"Module not found" error**: Run `pip install -r requirements.txt` again
2. **"API key invalid" error**: Check your `.env` file has the correct key
3. **"System Offline"**: Click "🚀 Start System" first
4. **Slow responses**: Ollama is slower than cloud APIs - this is normal

</details>

<details>
<summary><b>Can I use this for my business?</b></summary>

Yes! ASTRO is MIT licensed, meaning you can:
- ✅ Use it commercially
- ✅ Modify it for your needs
- ✅ Distribute it
- Just keep the license notice

</details>

<details>
<summary><b>How do I update ASTRO?</b></summary>

```bash
git pull origin main
pip install -r requirements.txt
```

Or download the latest ZIP from GitHub.

</details>

---

## 🏗️ How It Works (For the Curious)

```
                    ┌─────────────────────┐
                    │    Your Command     │
                    │  "Research AI..."   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Natural Language  │
                    │     Interpreter     │
                    │  (Understands you)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    Agent Engine     │
                    │ (The Brain/Manager) │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   ┌──────▼──────┐     ┌───────▼──────┐    ┌───────▼──────┐
   │  Research   │     │     Code     │    │  FileSystem  │
   │   Agent     │     │    Agent     │    │    Agent     │
   │ 🔍 Search   │     │ 💻 Program   │    │ 📁 Files     │
   └─────────────┘     └──────────────┘    └──────────────┘
```

**The magic happens in 4 steps:**

1. **You speak naturally** → "Find me information about..."
2. **AI understands** → Converts your words into tasks
3. **Agents work together** → Each specialist does their part
4. **Results delivered** → Files created, code run, answers found

---

## 🛠️ Advanced Configuration

### For Power Users

Edit `config/system_config.yaml`:

```yaml
system:
  environment: "production"    # or "development" for more logs
  log_level: "INFO"           # DEBUG, INFO, WARNING, ERROR
  max_concurrent_workflows: 10 # How many tasks at once

llm:
  provider: "openai"          # openai, ollama, openrouter
  model_name: "gpt-4"         # Your preferred model
  timeout: 60                 # Seconds to wait for response
```

### Agent-Specific Settings

Edit `config/agents.yaml`:

```yaml
research_agent_001:
  max_search_results: 10      # More results = more thorough
  max_pages_to_scrape: 5      # Pages to read in full

code_agent_001:
  safe_mode: true             # Keep this ON for security
  max_code_length: 10000      # Maximum code size

filesystem_agent_001:
  root_dir: "./workspace"     # Where files are saved
  allowed_extensions:         # What files can be created
    - .txt
    - .py
    - .md
    - .json
```

---

## 🧠 Enterprise Advanced Systems

ASTRO includes cutting-edge enterprise-grade systems for production deployments:

### 🔌 MCP (Model Context Protocol) Integration

Connect to external tools and resources via the standardized MCP protocol:

```python
from core.mcp_integration import MCPRegistry, MCPServerConfig

# Register an MCP server
registry = MCPRegistry()
await registry.register_server(MCPServerConfig(
    name="my_server",
    transport="http",
    url="http://localhost:3000"
))

# Call tools from any registered server
result = await registry.call_tool("search", {"query": "AI trends"})
```

**Features:**
- Multi-server support with automatic tool routing
- JSON-RPC 2.0 compliant protocol
- Retry logic with exponential backoff
- Tool discovery and capability negotiation

### 🤝 A2A (Agent-to-Agent) Protocol

Enable direct communication between agents using Google's A2A protocol:

```python
from core.a2a_protocol import A2ACoordinator, AgentCard, A2ATask

# Agents can discover and delegate tasks
coordinator = get_a2a_coordinator()
await coordinator.start()

# Request another agent to perform a task
task = A2ATask(task_id="001", name="research", input_data={"topic": "AI"})
capable_agent = coordinator.find_capable_agent(["research", "web_search"])
```

**Features:**
- Async message bus for agent communication
- Capability-based agent discovery
- Task delegation and collaboration
- Heartbeat and status monitoring

### 🏥 Self-Healing System

Automatic failure detection, recovery, and resilience:

```python
from core.self_healing import get_self_healing_system, CircuitBreaker

healing = get_self_healing_system()
await healing.start()

# Register components for monitoring
healing.register_component(
    "api_service",
    health_check=check_api_health,
    recovery_callback=restart_api
)

# Execute with full protection
success, result = await healing.execute_with_protection(
    "api_service", api_call, params
)
```

**Features:**
- Circuit breaker pattern (prevents cascade failures)
- Configurable retry policies with exponential backoff
- Health monitoring with latency tracking
- Automatic recovery strategies

### 🧮 Absolute Zero Reasoning Engine

First-principles reasoning without prior examples:

```python
from core.zero_reasoning import create_reasoner, ReasoningMode

reasoner = create_reasoner()

# Chain-of-thought reasoning
result = await reasoner.reason(
    "What are the implications of quantum computing for cryptography?",
    mode=ReasoningMode.DEDUCTIVE
)

# First-principles analysis
analysis = await reasoner.reason_from_first_principles(
    "How to design a scalable distributed system?"
)
```

**Features:**
- Chain-of-Thought (CoT) reasoning
- Tree-of-Thought (ToT) for complex problems
- Meta-cognitive self-evaluation
- Dynamic knowledge base with axioms

### 📚 Recursive Learning Framework

Continuous self-improvement through experience:

```python
from core.recursive_learning import get_recursive_learner, ExperienceType

learner = get_recursive_learner()

# Record experiences
learner.record_experience(
    experience_type=ExperienceType.TASK_COMPLETION,
    context={"task": "research"},
    action="web_search",
    outcome={"success": True},
    reward=0.9
)

# Learn from experiences
await learner.learn_batch(batch_size=32)

# Get action suggestions
action, confidence = learner.suggest_action({"task": "research"})
```

**Features:**
- Experience replay buffer with priority sampling
- Automatic pattern extraction
- Skill building from patterns
- Persistent knowledge storage

### 🔄 Refactory Feedback Loop

Automated code quality improvement:

```python
from core.refactory_loop import get_feedback_loop

loop = get_feedback_loop()

# Analyze and get suggestions
result = await loop.run_iteration(source_code, auto_apply=False)

print(f"Quality score: {result['before_score']:.2f} → {result['after_score']:.2f}")
print(f"Suggestions: {result['suggestion_count']}")
```

**Features:**
- Static code analysis (complexity, coverage, documentation)
- Automated refactoring suggestions
- Quality metrics tracking over time
- LLM-powered code transformation

### ⚡ Self-Adapting JIT Optimizer

Runtime optimization that adapts to usage patterns:

```python
from core.adaptive_jit import get_adaptive_jit, jit_profile, jit_memoize

jit = get_adaptive_jit()
await jit.start()

@jit.profile
def frequently_called(x):
    return expensive_computation(x)

@jit.memoize(ttl=300)
async def cached_api_call(query):
    return await api.search(query)
```

**Features:**
- Hot path detection
- Automatic memoization
- Adaptive caching strategies (LRU, LFU, TTL)
- Performance profiling and statistics

### 🧪 Running Enterprise Tests

```bash
# Run all advanced system tests
python tests/test_advanced_systems.py

# Expected output:
# ✅ ALL TESTS PASSED - SYSTEM IS PRODUCTION READY
```

You can also run the full health check:

```bash
python health_check.py
```

---

## ✅ Production Checklist

Before using ASTRO in a production-like environment, walk through this checklist:

1. **Install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. **Configure your AI provider**

   - Create a `.env` file with your API key(s) **or** use the in-app **Settings** dialog.
   - Supported options: **OpenAI**, **OpenRouter**, or local **Ollama**.

3. **Run automated checks**

   ```bash
   python -m pytest tests/test_advanced_systems.py
   python health_check.py
   ```

4. **Launch ASTRO (GUI)**

   ```bash
   python src/gui_app.py
   ```

---

---

## 📁 Project Structure

```
astro/
├── 📁 config/                 # Settings files
│   ├── system_config.yaml    # Main configuration
│   └── agents.yaml           # Agent-specific settings
│
├── 📁 src/                    # Source code
│   ├── 📁 agents/            # The AI workers
│   │   ├── research_agent.py # Searches the web
│   │   ├── code_agent.py     # Writes code
│   │   └── filesystem_agent.py # Manages files
│   │
│   ├── 📁 core/              # The brain + Enterprise systems
│   │   ├── engine.py         # Task manager
│   │   ├── nl_interface.py   # Understands you
│   │   ├── mcp_integration.py # MCP protocol support
│   │   ├── a2a_protocol.py   # Agent-to-Agent comms
│   │   ├── self_healing.py   # Fault tolerance
│   │   ├── zero_reasoning.py # First-principles AI
│   │   ├── recursive_learning.py # Self-improvement
│   │   ├── refactory_loop.py # Code quality
│   │   └── adaptive_jit.py   # Runtime optimization
│   │
│   ├── main.py               # Command-line version
│   └── gui_app.py            # Desktop app
│
├── 📁 workspace/              # Your files go here
├── 📁 tests/                  # Quality checks
├── requirements.txt          # Dependencies
└── README.md                 # You are here!
```

---

## 🆘 Getting Help

**Something not working?** Here's how to get help:

1. 📖 Check the [FAQ](#-frequently-asked-questions) above
2. 🔍 Search [existing issues](https://github.com/Senpai-Sama7/autonomous-agent-ecosystem/issues)
3. 🐛 [Report a bug](https://github.com/Senpai-Sama7/autonomous-agent-ecosystem/issues/new)
4. 💬 [Start a discussion](https://github.com/Senpai-Sama7/autonomous-agent-ecosystem/discussions)

---

## 🤝 Contributing

We love contributions! Whether it's:

- 🐛 Bug fixes
- ✨ New features
- 📖 Documentation improvements
- 🎨 UI enhancements

See our [Contributing Guide](CONTRIBUTING.md) to get started.

---

## 📄 License

MIT License - Use it freely, modify it, share it!

---

<div align="center">

### Made with ❤️ by the ASTRO Team

**[⬆ Back to Top](#-astro---autonomous-agent-ecosystem)**

</div>
