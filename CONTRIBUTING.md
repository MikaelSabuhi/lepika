# Contributing to LePika

Thanks for helping. LePika is small on purpose: a thin layer of Python that drives
Ollama, OpenWebUI, and `docker compose`. Read [`docs/architecture.md`](docs/architecture.md)
first; it lists the design rules a change has to keep.

## Setup

Only git and [uv](https://docs.astral.sh/uv/getting-started/installation/) are needed.

```sh
git clone https://github.com/MikaelSabuhi/lepika
cd lepika
uv sync --dev
uv run pre-commit install
uv run pytest -q
```

The test suite runs in about a second with no network, no Docker, and no real
processes, because every external effect is an injected callable. New code follows
the same pattern.

## Making a change

1. Open an issue first for anything bigger than a small fix, so we agree on the shape
   before you write it.
2. Work on a branch; `main` only changes through pull requests.
3. Write the failing test, then the code.
4. Run the gate before you push. It is the same one CI runs:

   ```sh
   uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q && uv run bandit -c pyproject.toml -r src -q && uv run pip-audit
   ```

5. Open the PR. The title is a conventional commit (`feat:`, `fix:`, `docs:`, `test:`,
   `ci:`, `chore:`); PRs are squash-merged, so that title becomes the commit on `main`.
   Keep each PR to one change.

## Rules worth knowing before you start

- **Three runtime dependencies** (typer, rich, structlog). Use the standard library for
  everything else; adding a dependency needs a discussion first.
- **Every user-facing failure is a `FriendlyError`**: one line saying what went wrong,
  one line saying what to do. No tracebacks reach the user.
- **Shell out, don't reimplement.** If Ollama, OpenWebUI, uv, or Docker already do it,
  LePika calls them.
- **Secrets never touch argv or logs.** Generated keys go in `0600` files.
- **The README is part of the product.** A change to a command, an install line, or the
  platform table updates it in the same PR.

## Reporting a bug

Use the bug report template. `lepika doctor` and `lepika logs` output is what makes a
report actionable, so please include both.

Security problems go through private reporting instead of an issue — see
[SECURITY.md](SECURITY.md).
