# Testing Guide

This document describes the testing strategy and how to run tests for the RAG system.

## Test Structure

```
rag_arxiv_qa/
├── tests/
│   ├── test_retrieval.py    # Retrieval component tests
│   └── ...
test_rag_service.py          # End-to-end RAG service tests
verify_setup.py              # Setup verification script
```

## Running Tests

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=rag_arxiv_qa --cov-report=html

# Run specific test file
pytest tests/test_retrieval.py -v
```

### Integration Tests

```bash
# Test RAG service end-to-end
python test_rag_service.py
```

### Setup Verification

```bash
# Verify environment and dependencies
python verify_setup.py
```

## CI/CD Testing

Tests run automatically on:
- Push to `main` or `dev` branches
- Pull requests

See `.github/workflows/ci.yml` for CI configuration.

## Test Coverage

Current test coverage includes:
- Retrieval component
- Embedding generation
- Vector store operations
- RAG service integration

Coverage reports are generated in CI and can be viewed locally:
```bash
pytest --cov=rag_arxiv_qa --cov-report=html
open htmlcov/index.html  # View coverage report
```

## Writing New Tests

When adding new features:
1. Add unit tests in `tests/` directory
2. Follow naming convention: `test_*.py`
3. Use pytest fixtures for setup/teardown
4. Mock external dependencies (LLM APIs, vector DB)

Example test structure:
```python
import pytest
from rag_arxiv_qa.src.retrieval.retriever import DenseRetriever

def test_retrieval_basic():
    # Test implementation
    pass
```
