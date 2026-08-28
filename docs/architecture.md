# Architecture

How ezai works, and why it's built this way. For what it does, read the [README](../README.md).

## The shape of it

ezai is a thin orchestrator. It never reimplements inference, UIs, or package management — it detects the machine, asks a few questions, and drives the real tools:

```
ezai CLI (Python, Typer + Rich)
 ├─ detect.py   OS / arch / GPU / RAM / what's already installed
 ├─ wizard.py   the default `ezai` experience (≤4 questions)
 ├─ models.py   model refs (3 shapes) + curated list (models.toml)
 ├─ express.py  process lifecycle: native Ollama + OpenWebUI via uv
 ├─ config.py   ~/.ezai/config.toml (flat, versioned, atomic writes)
 ├─ doctor.py   diagnostics; every ✗ ships a one-line fix
 ├─ proc.py     the single subprocess choke point (logged, friendly)
 └─ cli.py      commands: up · down · status · logs · model · doctor · update
State: ~/.ezai/ (override with EZAI_HOME)
```

**Express mode** (v0.1, the default): no Docker anywhere. Ollama runs natively (Homebrew / official script / winget) so the GPU works on every platform — Metal on Mac, CUDA on Windows/Linux — and OpenWebUI runs via `uv tool`. **Server mode** (planned): docker compose stack with an ollama container, optional vLLM (Linux + NVIDIA), and Caddy-fronted API-key exposure for the network.

## Design rules

These are load-bearing; changes should preserve them.

1. **Two runtime dependencies** — typer and rich. Stdlib for everything else (urllib, tomllib, ctypes, socket).
2. **Every user-reachable failure is a `FriendlyError(problem, fix)`** — one red line, one suggested next step, never a traceback. `proc.run_logged` converts nonzero exits, missing binaries, and timeouts; everything else raises it deliberately.
3. **Every external effect is an injected callable** (`run`, `which`, `popen`, `urlopen`, `sleep`, `call`, `kill`, `bind`). The test suite (111 tests) runs in under a second with no network, no Docker, and no real processes — and an autouse fixture points `EZAI_HOME` at a temp dir so tests can never touch a real `~/.ezai`.
4. **Model refs, one field, three shapes:** `qwen3:8b` (Ollama tag) · `hf.co/<org>/<repo>-GGUF` (Ollama pulls from Hugging Face) · `<org>/<repo>` (full weights → vLLM, Server mode; rejected today with a GGUF hint).
5. **The curated list is data, not code.** `models.toml` ships in the wheel and is re-fetched from `main` at runtime (3s timeout). Remote content is untrusted: unknown keys are dropped, bad types and missing fields skip the entry, parse errors fall back to the bundled copy. The bundled copy parses strictly so defects fail in CI.
6. **Health-check first, act second.** A healthy `ezai up` makes zero subprocess calls and works offline. Restarts wait for the old server to actually die before declaring victory.
7. **pid files are hints, not truth.** A pid is only signalled if the webui is actually answering on the configured port — a stale file after a reboot can never kill an unrelated process.

## Notes for maintainers

- **PyPI:** the name `ezai` on PyPI belongs to an unrelated package. Installers therefore install from this repository (`git+https://…`) and must never install a bare name we don't own. Publishing requires choosing a distribution name first.
- **Ports:** conflict detection probes both `0.0.0.0` and `127.0.0.1` (macOS lets loopback bind over a wildcard listener) and uses `SO_EXCLUSIVEADDRUSE` on Windows.
- **`ollama` and `open-webui` track latest** by design; `ezai update` is the refresh path. Breakage from upstream is caught by tests + (planned) scheduled smoke runs.
- **Known trade-off:** `ezai down` treats a hung-but-alive webui as stale (won't signal it). Follow-up idea: pid liveness fallback.
- Remote engines (`ezai connect`), network exposure with API keys (`ezai expose`), vLLM, and the compose stack are Server-mode scope — planned, not present. `config.engine_url` is already honored by status/doctor/engine checks in preparation.
