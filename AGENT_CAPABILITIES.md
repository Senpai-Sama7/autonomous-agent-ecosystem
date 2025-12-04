# ASTRO Agent Capabilities

Four new agents have been added to enhance ASTRO's agentic capabilities:

## 1. Git Agent (`git_agent_001`)
**Purpose**: Version control mastery and repository management

**Operations**:
- `status` - Show git status
- `diff` - Show file differences
- `add` - Stage files for commit
- `commit` - Create commits with messages
- `checkout` - Switch branches
- `branch` - Manage branches
- `log` - View commit history

**Example**:
```python
await toolkit.git_ops("status")
await toolkit.git_ops("commit", ["--message", "Fix bug in parser"])
await toolkit.git_ops("checkout", ["-b", "feature/new-feature"])
```

---

## 2. Test Agent (`test_agent_001`)
**Purpose**: Test-driven development and quality assurance

**Operations**:
- Run any test command (pytest, npm test, cargo test, etc.)
- Optional target file for focused testing
- Returns test output, errors, and exit code

**Example**:
```python
await toolkit.run_test("pytest", "tests/test_parser.py")
await toolkit.run_test("npm test")
await toolkit.run_test("cargo test")
```

---

## 3. Analysis Agent (`analysis_agent_001`)
**Purpose**: Static code analysis and linting

**Operations**:
- Run linting tools (pylint, eslint, cargo check, etc.)
- Analyze specific files or directories
- Returns analysis output and warnings

**Example**:
```python
await toolkit.code_analysis("pylint", "src/")
await toolkit.code_analysis("eslint", "src/app.js")
await toolkit.code_analysis("cargo check", ".")
```

---

## 4. Knowledge Agent (`knowledge_agent_001`)
**Purpose**: Semantic memory and architectural context persistence

**Operations**:
- `save` - Persist knowledge with a key
- `retrieve` - Recall saved knowledge

**Example**:
```python
# Save architectural decision
await toolkit.knowledge_manager("save", "project_architecture", 
    "Microservices with async message queue")

# Retrieve for future sessions
result = await toolkit.knowledge_manager("retrieve", "project_architecture")
```

---

## Integration with LLM

These agents are now available to the LLM through the `AgentToolkit`. The system prompt should include:

> "You have access to specialized agents for git operations, testing, code analysis, and knowledge management. Use these tools to verify your work, maintain version control, and persist important context."

---

## Workflow Example: Complete Development Cycle

```
1. User: "Add a new feature to parse JSON files"
   ↓
2. Code Agent: Writes parser code
   ↓
3. Analysis Agent: Runs pylint on new code
   ↓
4. Test Agent: Runs pytest to verify functionality
   ↓
5. Git Agent: Commits changes with semantic message
   ↓
6. Knowledge Agent: Saves "JSON parser implementation" for future reference
   ↓
7. Result: Feature complete, tested, analyzed, and documented
```

---

## File Locations

- **Agents**: `src/agents/`
  - `git_agent.py`
  - `test_agent.py`
  - `analysis_agent.py`
  - `knowledge_agent.py`

- **Integration**: `src/core/`
  - `agent_registry.py` - Initialization
  - `agent_tools.py` - LLM-facing toolkit

- **Knowledge Storage**: `workspace/knowledge/` (auto-created)

---

## Next Steps

1. Import `AgentToolkit` in your engine initialization
2. Call `initialize_agents(engine)` to register all agents
3. Pass toolkit to LLM context for tool availability
4. Update system prompt to reference these capabilities
