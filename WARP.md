# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Environment & setup

- **Python:** 3.10+.
- From the repo root, create and activate a virtualenv, then install dependencies:

```bash
python -m venv venv
# Windows PowerShell: .\venv\Scripts\Activate.ps1
# Unix/macOS: source venv/bin/activate
pip install -r requirements.txt
```

All runtime and tooling dependencies are pinned in `requirements.txt` for supply-chain security.

## Core commands

### Health & validation

These are the main entry points to check that the system is "production ready" before making intrusive changes:

- **Comprehensive health check** (network, DB, configs, integration tests, GUI imports, optional LLM + web search):

  ```bash
  python health_check.py
  ```

  Required checks: database, config files, pytest suite, GUI imports. Optional checks: network, Ollama server, LLM end‑to‑end, web search.

- **Production validation script** (imports, DB, FileSystemAgent, configs, dependencies):

  ```bash
  python tests/validate_production.py
  ```

  Exits non‑zero if any of the checks fail.

### Tests

- **Run the full test suite** (unit + integration):

  ```bash
  python -m pytest tests/ -v
  ```

- **Run only the advanced systems integration suite** (`MCP`, `A2A`, self‑healing, reasoning, learning, refactory loop, adaptive JIT):

  ```bash
  python -m pytest tests/test_advanced_systems.py -v
  ```

- **Run a single advanced systems test** (example: MCP integration only):

  ```bash
  python -m pytest tests/test_advanced_systems.py::test_mcp_integration -v
  ```

### Linting & quality gates

Quality gates are defined in `docs/quality-gates.md` and enforced manually / in CI:

- **Syntax check (no Python files with syntax errors):**

  ```bash
  python -m py_compile src/**/*.py
  ```

- **Type checking (when mypy is available/enabled):**

  ```bash
  mypy src/ --ignore-missing-imports
  ```

- **Secret scan (hard‑coded API keys):**

  ```bash
  grep -rn "sk-\|api_key.*=.*['\"]" src/ --include="*.py" | grep -v "getenv\|environ"
  ```

- **Pre‑commit hooks** (whitespace, YAML/JSON, large files, key detection, and project‑specific hooks to block imports from `.trash` and tracked artifacts):

  ```bash
  pip install pre-commit
  pre-commit install
  pre-commit run --all-files
  ```

### Running the application

- **GUI application (recommended end‑user entry point):**

  ```bash
  python src/gui_app.py
  ```

- **CLI main in workflow demo mode** (creates sample workflows and runs engine + monitoring for a fixed duration):

  ```bash
  python src/main.py --duration 120
  ```

- **CLI in interactive natural‑language mode** (no sample workflows; user drives the system via NL requests):

  ```bash
  python src/main.py --interactive \
      --llm-provider openai|openrouter|ollama \
      --model-name gpt-3.5-turbo \
      [--api-key YOUR_KEY] \
      [--api-base CUSTOM_BASE_URL] \
      [--debug]
  ```

  In interactive mode `NaturalLanguageInterface` turns each prompt into one or more `Workflow` objects and submits them to the `AgentEngine`.

### Build artifacts

These scripts are POSIX shell oriented; run from a bash‑compatible shell with the virtualenv active.

- **PyInstaller executable (Windows/Linux):**

  ```bash
  bash build_exe.sh
  ```

  Produces `dist/ASTRO.exe` (Windows) or `dist/ASTRO` (Linux) using `astro.spec`.

- **Debian package:**

  ```bash
  bash build_deb.sh
  ```

  Produces `astro_1.0.0_amd64.deb` from the `debian/` layout.

## High‑level architecture

### Overview

ASTRO is a production‑grade multi‑agent system that turns natural‑language requests into orchestrated workflows executed by specialized agents. At a high level:

- **User interfaces** (GUI and CLI) accept natural‑language requests and display logs, reasoning traces, and workflow history.
- **Natural Language Interface** converts user input into structured workflows and tasks.
- **Agent Engine** schedules tasks, routes them to the right agents, persists state, and tracks performance.
- **Specialized agents** perform web research, code generation/execution, and filesystem operations.
- **Enterprise systems** provide MCP tooling, agent‑to‑agent communication, self‑healing, structured reasoning, recursive learning, refactory feedback loop, and adaptive caching.
- **Observability layer** (monitoring dashboard + health checks) exposes metrics, health summaries, and alerting signals.

### Entry points & user interfaces

- **GUI (`src/gui_app.py`)**
  - CustomTkinter‑based glassmorphic desktop UI.
  - Wires together `AgentEngine`, `MonitoringDashboard`, `LLMFactory`, `NaturalLanguageInterface`, and app state utilities.
  - Presents:
    - Command input box for plain‑language requests.
    - Agent status cards for `research_agent_001`, `code_agent_001`, and `filesystem_agent_001`.
    - Reasoning/logs panel showing chain‑of‑thought style traces and system logs.
    - Workflow/task history panel.
    - Settings dialog for choosing provider/model and setting API keys.
    - An onboarding tutorial driven by `TUTORIAL_STEPS`.

- **CLI (`src/main.py`)**
  - Defines `AutonomousAgentEcosystem`, which owns an `AgentEngine`, `MonitoringDashboard`, agent instances, and registered workflows.
  - Command‑line flags (parsed in `parse_arguments()`):
    - `--duration` runtime for non‑interactive demo mode.
    - `--interactive` to enter REPL‑style NL mode.
    - `--debug` to enable debug logging.
    - `--llm-provider`, `--model-name`, `--api-key`, `--api-base` to configure the LLM layer.
  - In non‑interactive mode, creates a sample ML‑oriented workflow (`workflow_ml_001`) with dependent tasks flowing through the research, code‑gen, and execution phases.
  - In interactive mode, wraps the engine with `NaturalLanguageInterface` and keeps reading user input in a loop, turning each request into a workflow.

### Core orchestration layer (`src/core`)

- **Engine (`core/engine.py`)**
  - Central scheduler and orchestrator.
  - Key types:
    - `AgentConfig` – capabilities, concurrency limits, reliability, cost per operation.
    - `Task` – unit of work with required capabilities, priority, deadline, dependencies, payload, and retry tracking.
    - `Workflow` – a set of `Task` instances with shared priority, execution budget, and success criteria.
  - Maintains:
    - `agents` (logical configs) and `agent_instances` (actual Python objects).
    - A `PriorityQueue` of pending tasks; priority score derived from workflow priority, deadlines, and dependencies.
    - `completed_tasks` / `failed_tasks` sets for dependency tracking.
    - Per‑agent performance metrics and status (`AgentStatus`).
  - Persists agents, workflows, and tasks asynchronously via `core.database.DatabaseManager` into `ecosystem.db`.
  - Main loop (`start_engine`) pulls tasks, checks dependency satisfaction, selects a suitable agent (capabilities + status), and delegates execution, with basic retry logic for transient errors.

- **Database (`core/database.py`)**
  - Encapsulates all SQLite access and schema management.
  - Provides async/await‑friendly helpers used by the engine and monitoring to store agents, workflows, tasks, and metrics.

- **Natural Language Interface (`core/nl_interface.py`)**
  - Translates free‑form text into structured workflows and tasks, using an LLM client from `LLMFactory`.
  - Responsible for task decomposition, capability tagging, and submitting work to the engine.

- **LLM factory (`core/llm_factory.py`)**
  - Single place to construct async and sync LLM clients for supported providers: OpenAI, OpenRouter, Ollama.
  - Reads provider/model configuration from CLI args and/or `config/system_config.yaml`.
  - Used by the main app, GUI, and agents (especially `CodeAgent` and `ResearchAgent`).

- **Monitoring & observability (`src/monitoring/monitoring_dashboard.py`)**
  - Aggregates task, agent, and system metrics.
  - Exposes `get_system_health()`, `get_agent_performance()`, and `get_alert_summary()` used by `AutonomousAgentEcosystem.generate_final_report()` and docs in `docs/observability.md`.

- **Enterprise subsystems (advanced features)**
  - `core/mcp_integration.py` – MCP client/registry implementing JSON‑RPC over HTTP, capability discovery, and tool/resource registration.
  - `core/a2a_protocol.py` – Agent‑to‑Agent protocol implementation:
    - `AgentCard` capability metadata, `A2AMessage` envelopes, `A2ATask` structures.
    - `MessageBus` async message bus for inter‑agent messaging.
    - Coordinator helper (`get_a2a_coordinator`) for discovery and delegation.
  - `core/self_healing.py` – resilience primitives:
    - `CircuitBreaker`, `RetryPolicy`, `HealthMonitor`, `RecoveryManager`, and a singleton self‑healing system.
  - `core/zero_reasoning.py` – structured reasoning engine abstractions (chain‑of‑thought/tree‑of‑thought, knowledge base, meta‑cognition).
  - `core/recursive_learning.py` – experience store and suggestion engine for task‑level learning over time.
  - `core/refactory_loop.py` – code quality analysis and refactoring feedback loop.
  - `core/adaptive_jit.py` – adaptive caching and hot‑path profiling (memoization, not byte‑code JIT).

### Agents layer (`src/agents`)

- **Base abstractions (`agents/base_agent.py`)**
  - `AgentCapability`, `AgentState`, `AgentContext`, and `TaskResult` define the common contract.
  - `BaseAgent` tracks task history, computes success rate, exposes `health_check()` and `self_diagnose()`, and provides a recovery hook.

- **ResearchAgent (`agents/research_agent.py`)**
  - Performs web search via `ddgs` (DuckDuckGo) and HTTP requests, parses results with BeautifulSoup, and synthesizes summaries via LLMs.
  - Tuned via `config/agents.yaml` (`max_search_results`, `max_scrape_results`, preferred domains, concurrency, quality thresholds).

- **CodeAgent (`agents/code_agent.py`)**
  - Handles code generation, analysis, and (optionally sandboxed) execution.
  - Uses `LLMFactory` to talk to the configured provider/model.
  - Enforces security policy described in the README and engineering charter:
    - AST + regex‑based validation in `safe_mode`.
    - Docker sandboxing controlled by `use_docker_sandbox`, `allow_local_execution`, `docker_image`, and `docker_execution_timeout` in `config/agents.yaml`.
    - File extension allow‑list for generated artifacts.

- **FileSystemAgent (`agents/filesystem_agent.py`)**
  - Handles all file I/O for user‑visible artifacts.
  - Root directory and allowed extensions are strictly controlled by `filesystem_agent_001` config in `config/agents.yaml` (default root: `./workspace`).
  - Uses path‑normalization and `os.path.commonpath` style checks to prevent writing outside the workspace.

### Configuration & conventions

- **System configuration (`config/system_config.yaml`)**
  - `system.environment`, `log_level`, `max_concurrent_workflows`, `enable_monitoring` control global runtime behavior.
  - `llm.provider`, `model_name`, `timeout`, `max_retries` define default LLM behavior; can be overridden via CLI flags.

- **Agent configuration (`config/agents.yaml`)**
  - Per‑agent operational settings and safety policies.
  - Notable production defaults:
    - `code_agent_001.use_docker_sandbox: true`
    - `code_agent_001.allow_local_execution: false`
    - `filesystem_agent_001.root_dir: "./workspace"` and a small allow‑listed set of extensions (`.txt`, `.py`, `.md`, `.json`, `.csv`, `.log`, `.yaml`).

- **Workspace & database**
  - User‑generated files are meant to live under `workspace/` by default.
  - Persistent system state (agents, workflows, tasks, metrics) lives in `ecosystem.db` at the repo root; tests and health checks assume this file and its schema.

### Tests & quality documentation

- **Integration tests (`tests/test_advanced_systems.py`)**
  - One async test per advanced subsystem (MCP, A2A, self‑healing, zero reasoning, recursive learning, refactory loop, adaptive JIT).
  - Exposed in README and docs as the canonical integration suite; invoked by `health_check.py` and `docs/quality-gates.md`.

- **Production validation (`tests/validate_production.py`)**
  - Standalone script that verifies imports, database operations, FileSystemAgent basic behavior, configuration validity, dependency presence, and engine initialization.

- **Docs**
  - `docs/engineering-charter.md` – core principles (production‑grade quality, security, observability, maintainability) and system architecture diagram.
  - `docs/quality-gates.md` – concrete pre‑commit, CI, and release gates plus validation command examples.
  - `docs/observability.md` – logging strategy, metrics taxonomy, health check behavior, and alerting guidance.

## How future agents should use this

- Prefer `python health_check.py` and `python -m pytest tests/ -v` before and after substantial changes, especially in `src/core` or `src/agents`.
- When adding or modifying agents, keep their configuration in `config/agents.yaml` in sync and respect the existing security model (sandboxing, path restrictions).
- When wiring new features into the UI, go through `AutonomousAgentEcosystem` / `AgentEngine` rather than bypassing the workflow system, so monitoring and persistence stay consistent.
