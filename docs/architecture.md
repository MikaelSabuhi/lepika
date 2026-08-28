# Architecture

How LePika works, and why it's built this way. For what it does, read the [README](../README.md).

## The shape of it

LePika is a thin orchestrator. It never reimplements inference, UIs, or package management — it detects the machine, asks a few questions, and drives the real tools:

```
LePika CLI (Python, Typer + Rich)
 ├─ detect.py   OS / arch / GPU / RAM / what's already installed
 ├─ wizard.py   the default `lepika` experience (≤4 questions)
 ├─ models.py   model refs (3 shapes) + curated list (models.toml)
 ├─ express.py  process lifecycle: native Ollama + OpenWebUI via uv
 ├─ config.py   ~/.lepika/config.toml (flat, versioned, atomic writes)
 ├─ doctor.py   diagnostics; every ✗ ships a one-line fix
 ├─ proc.py     the single subprocess choke point (logged, friendly)
 ├─ log.py      JSON-lines log, secret-looking keys redacted
 └─ cli.py      commands: up · down · status · logs · model · doctor · update
State: ~/.lepika/ (override with LEPIKA_HOME)
```

**Express mode** (v0.1, the default): no Docker anywhere. Ollama runs natively (Homebrew / official script / winget) so the GPU works on every platform — Metal on Mac, CUDA on Windows/Linux — and OpenWebUI runs via `uv tool`. **Server mode** (planned): docker compose stack with an ollama container, optional vLLM (Linux + NVIDIA), and Caddy-fronted API-key exposure for the network.

## Design rules

These are load-bearing; changes should preserve them.

1. **Three runtime dependencies** — typer, rich, and structlog (JSON-lines log under `~/.lepika/logs/lepika.log`; keys named like secrets are redacted before they are written). Events are `area.action`; a line is written when something changed or failed, never for a pure read (`run_logged(..., log=False)`) and never for a health probe. Stdlib for everything else (urllib, tomllib, ctypes, socket, json, secrets).
2. **Every user-reachable failure is a `FriendlyError(problem, fix)`** — one red line, one suggested next step, never a traceback. `proc.run_logged` converts nonzero exits, missing binaries, and timeouts; everything else raises it deliberately.
3. **Every external effect is an injected callable** (`run`, `which`, `popen`, `urlopen`, `sleep`, `call`, `kill`, `bind`). The test suite (118 tests) runs in under a second with no network, no Docker, and no real processes — and an autouse fixture points `LEPIKA_HOME` at a temp dir so tests can never touch a real `~/.lepika`.
4. **Model refs, one field, three shapes:** `qwen3:8b` (Ollama tag) · `hf.co/<org>/<repo>-GGUF` (Ollama pulls from Hugging Face) · `<org>/<repo>` (full weights → vLLM, Server mode; rejected today with a GGUF hint).
5. **The curated list is data, not code.** `models.toml` ships in the wheel and is re-fetched from `main` at runtime (3s timeout). Remote content is untrusted: unknown keys are dropped, bad types and missing fields skip the entry, parse errors fall back to the bundled copy. The bundled copy parses strictly so defects fail in CI.
6. **Health-check first, act second.** A healthy `lepika up` makes zero subprocess calls and works offline. Restarts wait for the old server to actually die before declaring victory.
7. **pid files are hints, not truth.** A pid is only signalled if the webui is actually answering on the configured port — a stale file after a reboot can never kill an unrelated process.

## Notes for maintainers

- **PyPI:** LePika is not published to PyPI yet. Installers therefore install from this repository (`git+https://…`) and must never install a bare name we don't own. Publishing requires registering the distribution name first.
- **Ports:** conflict detection probes both `0.0.0.0` and `127.0.0.1` (macOS lets loopback bind over a wildcard listener) and uses `SO_EXCLUSIVEADDRUSE` on Windows.
- **`ollama` and `open-webui` track latest** by design; `lepika update` is the refresh path. Breakage from upstream is caught by tests + (planned) scheduled smoke runs.
- **Known trade-off:** `lepika down` treats a hung-but-alive webui as stale (won't signal it). Follow-up idea: pid liveness fallback.
- Remote engines (`lepika connect`), network exposure with API keys (`lepika expose`), vLLM, and the compose stack are Server-mode scope — planned, not present. `config.engine_url` is already honored by status/doctor/engine checks in preparation.
