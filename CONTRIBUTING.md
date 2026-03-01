# Contributing to EmpathEase

Thank you for your interest in contributing! This document outlines the process for contributing to EmpathEase.

## Getting Started

1. Fork the repository and clone your fork
2. Follow the [Quick Start](README.md#-quick-start) guide to set up your local environment
3. Create a feature branch: `git checkout -b feat/your-feature-name`

## Branch Naming

| Prefix | Use for |
|---|---|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `refactor/` | Code refactoring |
| `test/` | Adding or updating tests |
| `chore/` | Maintenance tasks |

## Commit Style

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(fusion): add confidence decay for stale emotion frames
fix(vocal): handle silent audio segments gracefully
docs(readme): update environment variable table
```

## Pull Request Process

1. Ensure all tests pass: `pytest tests/ -v`
2. Add or update tests for any new behaviour
3. Update `README.md` if you change any public API or configuration
4. Open a PR against `main` with a clear description of what changed and why

## Code Style

- **Python**: Follow PEP 8. We recommend `ruff` for linting
- **JavaScript/React**: Standard ESLint defaults (Vite preset)
- Keep functions small and focused; add docstrings to public functions

## Sensitive Data

- Never commit `.env` files or API keys
- ML model weights (`.onnx`, `.pt`, `.pth`) are excluded from the repo — see the model download instructions in `backend/models/`

## Reporting Issues

Please use the GitHub Issues templates provided in `.github/ISSUE_TEMPLATE/`. Include:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behaviour
- Relevant logs or screenshots
