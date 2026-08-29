## What

<!-- One or two sentences: what changes, and why. Link the issue if there is one. -->

Closes #

## Checklist

- [ ] Title is a conventional commit (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`)
- [ ] Tests cover the change (the failing test came first)
- [ ] The gate passes locally: `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q && uv run bandit -c pyproject.toml -r src -q`
- [ ] README updated if a command, install line, or the platform table changed
- [ ] No new runtime dependency (LePika has three: typer, rich, structlog)
