# Agentic Document Extraction – Python Library

Agentic Document Extraction is a Python library that wraps LandingAI's Vision Agent API for structured data extraction from complex documents (PDFs, images, charts, tables). The library provides long-document support, auto-retry mechanisms, parallel processing, and visual debugging utilities.

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Bootstrap, Build, and Test the Repository
- Install Poetry 1.4.2: `pip install poetry==1.4.2`
- Configure Poetry: `poetry config virtualenvs.in-project true`
- Install dependencies: `poetry install --all-extras` -- takes 40 seconds to complete. **NEVER CANCEL.** Set timeout to 120+ seconds.
- Verify installation: `poetry env info` and `poetry run python -c "from agentic_doc.parse import parse; print('Import successful')"`

### Linting and Code Quality
- **NEVER CANCEL builds or long-running commands** - Always wait for completion
- Linting: `poetry run flake8 . --exclude .venv,examples,tests --count --show-source --statistics` -- takes <2 seconds
- Format checking: `poetry run black --check --diff --color agentic_doc/` -- takes <2 seconds
- Type checking: `poetry run mypy agentic_doc` -- takes 15 seconds. **NEVER CANCEL.** Set timeout to 30+ seconds.

### Testing
- Unit tests: `poetry run pytest -s -vvv tests/unit` -- takes 10 seconds. **NEVER CANCEL.** Set timeout to 30+ seconds.
- Integration tests: `poetry run pytest -n auto -s -vvv tests/integ` -- **requires VISION_AGENT_API_KEY environment variable**. Takes 60+ seconds. **NEVER CANCEL.** Set timeout to 120+ seconds.
- All tests: `poetry run pytest` -- combines unit and integration tests

### API Key Setup
- **Required for integration tests and actual functionality**: Set `VISION_AGENT_API_KEY=<your-api-key>` environment variable
- Get API key from: https://va.landing.ai/settings/api-key
- Library will load settings and log configuration on import (this is normal behavior)

## Validation

### Manual Testing Scenarios
**CRITICAL:** Since this is a library without a UI, validation focuses on import functionality and error handling:

1. **Basic Import Test**: 
   ```bash
   poetry run python -c "from agentic_doc.parse import parse; print('✓ Import successful')"
   ```
   Expected: Settings log appears, then "✓ Import successful"

2. **Configuration Loading Test**:
   ```bash
   poetry run python -c "from agentic_doc.config import Settings; s = Settings(); print('✓ Config loaded - batch size:', s.batch_size)"
   ```
   Expected: Settings log, then "✓ Config loaded - batch size: 4"

3. **Without API Key Test** (should work for imports):
   ```bash
   unset VISION_AGENT_API_KEY && poetry run python -c "from agentic_doc.parse import parse; print('✓ Package structure valid')"
   ```
   Expected: API key shown as "[REDACTED]", but import succeeds

4. **Code Quality Pipeline**:
   ```bash
   poetry run flake8 . --exclude .venv,examples,tests --count --show-source --statistics &&
   poetry run black --check --diff --color agentic_doc/ &&
   poetry run mypy agentic_doc &&
   echo "✓ All code quality checks passed"
   ```

### Build Validation
- **Always run the complete dependency installation sequence**: Poetry install can take 40+ seconds and installs 110+ packages
- **Always run all code quality checks** before committing: flake8, black, mypy
- **Always run unit tests** before committing: `poetry run pytest tests/unit`
- Integration tests require API key and network access - skip if not available

## Common Tasks

### Adding Dependencies
- Add to pyproject.toml under appropriate section ([tool.poetry.dependencies] or [tool.poetry.group.dev.dependencies])
- Run: `poetry install --all-extras` -- **NEVER CANCEL.** Allow 60+ seconds.

### Code Formatting
- Auto-format: `poetry run black agentic_doc/`
- Check format: `poetry run black --check --diff --color agentic_doc/`

### Running Specific Tests
```bash
# Single test file
poetry run pytest tests/unit/test_parse.py

# Specific test
poetry run pytest tests/unit/test_parse.py::TestParseDocument::test_parse_single_page_pdf

# With coverage
poetry run pytest --cov=agentic_doc tests/unit/
```

### Environment Configuration
Set environment variables in .env file:
```bash
# API access
VISION_AGENT_API_KEY=your-api-key-here

# Performance tuning
BATCH_SIZE=4                    # Files processed in parallel
MAX_WORKERS=5                   # Threads per file
MAX_RETRIES=3                   # Retry attempts
MAX_RETRY_WAIT_TIME=60         # Max wait between retries
```

## Repository Structure Reference

### Key Files and Directories
```
agentic_doc/          # Main package code
├── parse.py          # Main parsing functionality
├── config.py         # Configuration and settings
├── common.py         # Data models and types
├── connectors.py     # File source connectors (S3, Google Drive, etc.)
└── utils.py          # Utility functions

tests/                # Test suite
├── unit/             # Unit tests (no API calls)
├── integ/            # Integration tests (requires API key)
└── conftest.py       # Test fixtures

.github/workflows/    # CI/CD pipelines
├── ci-unit.yml       # Unit test workflow
└── ci-integ.yml      # Integration test workflow
```

### Configuration Files
- `pyproject.toml`: Poetry dependencies, build config, tool settings
- `.flake8`: Linting configuration (E501, E203 ignored, max-line-length=88)
- `poetry.lock`: Locked dependency versions

### Key Package Functions
- `parse()`: Main function for document extraction
- `parse_documents()`: Legacy batch parsing function  
- `viz_parsed_document()`: Visualization utility
- Connectors: LocalConnector, S3Connector, GoogleDriveConnector, URLConnector

## Timing Expectations

**CRITICAL - NEVER CANCEL these operations:**
- **Dependency installation**: 40 seconds (120+ packages) - Set timeout to 120+ seconds
- **Type checking**: 15 seconds - Set timeout to 30+ seconds  
- **Unit tests**: 10 seconds - Set timeout to 30+ seconds
- **Integration tests**: 60+ seconds (with API calls) - Set timeout to 120+ seconds
- **Linting**: <2 seconds each (flake8, black)
- **Import tests**: <2 seconds

## Troubleshooting

### Common Issues
- **Poetry not found**: Install with `pip install poetry==1.4.2`
- **Import errors after dependency changes**: Run `poetry install --all-extras` again
- **Type checking failures**: Check mypy configuration in pyproject.toml
- **Test failures without API key**: Normal for integration tests, unit tests should pass
- **Network errors in tests**: Expected in restricted environments, check unit vs integration test separation

### Required Environment
- Python 3.9, 3.10, 3.11, or 3.12
- Poetry 1.4.2 
- Network access for integration tests and actual API functionality
- VISION_AGENT_API_KEY for full functionality

### CI/CD Context
- Unit tests run on: Python 3.9, 3.13 × Ubuntu, Windows, macOS
- Integration tests run on: Ubuntu with Python 3.12
- Uses `poetry config virtualenvs.in-project true` in CI
- Parallel test execution with `pytest -n auto` for integration tests