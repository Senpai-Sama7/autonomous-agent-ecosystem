# Astro Agent Guidelines

## Build & Test Commands
- Install dependencies: `pip install -r requirements.txt -r requirements-dev.txt`
- Run all tests: `pytest`
- Run single test: `pytest tests/path/to/test_file.py::test_name`
- Lint code: `pre-commit run --all-files`
- Build DEB package: `./build_deb.sh`

## Code Style Rules
1. **Formatting**:
   - Follow Black formatting (line-length=120)
   - Imports: `isort` order (stdlib, third-party, local)
2. **Typing**:
   - Type hints required for all function signatures
   - Use `Optional` and `Union` for complex types
3. **Naming**:
   - Classes: `UpperCamelCase`
   - Functions/variables: `snake_case`
   - Constants: `ALL_CAPS`
4. **Error Handling**:
   - Use custom exceptions from `src/core/exceptions.py`
   - Never use bare `except:`
   - Log errors with context: `logger.error("Message", extra={"context": data})`

## Agent-Specific Notes
- Always validate inputs with `pydantic` models
- Use `@logger.catch` decorator for critical functions
- Write tests for all agent actions in `tests/agents/`