# Agentic-Doc Tests

This directory contains tests for the Agentic-Doc project.

## Test Structure

- `unit/`: Unit tests for individual components
- `integ/`: Integration tests that test multiple components together
- `conftest.py`: Global test fixtures and utilities

## Running Tests

To run the tests, first install the development requirements:

```bash
poetry install --all-extras
poetry shell
```

Then run the tests with pytest:

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run specific test file
pytest tests/unit/test_parse_document.py

# Run a specific test
pytest tests/unit/test_parse_document.py::TestParseAndSaveDocument::test_parse_single_page_pdf
```

## Adding New Tests

When adding new tests:

1. Place unit tests in the `unit/` directory
2. Place integration tests in the `integ/` directory
3. Add any needed fixtures to the relevant `conftest.py` file
4. Follow the existing patterns (Arrange-Act-Assert format) 