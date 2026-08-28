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
 ├─ server.py   Server mode: the docker compose stack under ~/.lepika/stack
 ├─ proc.py     the single subprocess choke point (logged, friendly)
 ├─ log.py      JSON-lines log, secret-looking keys redacted
 └─ cli.py      commands: up · down · status · logs · model · doctor · update · connect
State: ~/.lepika/ (override with LEPIKA_HOME)
```

**Express mode** (v0.1, the default): no Docker anywhere. Ollama runs natively (Homebrew / official script / winget) so the GPU works on every platform — Metal on Mac, CUDA on Windows/Linux — and OpenWebUI runs via `uv tool`. **Server mode** (`--mode server`, or picked in the wizard when Docker is already installed): docker compose stack with an ollama container, a Caddy proxy that gates the engine API on a generated key (`lepika expose`), plus a vLLM profile (Linux + NVIDIA) for full-weight Hugging Face repos. `cli._backend(cfg)` picks the module — `express` and `server` expose the same `start_stack`/`stop`/`update`/`logs` surface, so every lifecycle command is written once.

## Design rules

These are load-bearing; changes should preserve them.

1. **Three runtime dependencies** — typer, rich, and structlog (JSON-lines log under `~/.lepika/logs/lepika.log`; keys named like secrets are redacted before they are written). Events are `area.action`; a line is written when something changed or failed, never for a pure read (`run_logged(..., log=False)`) and never for a health probe. Stdlib for everything else (urllib, tomllib, ctypes, socket, json, secrets).
2. **Every user-reachable failure is a `FriendlyError(problem, fix)`** — one red line, one suggested next step, never a traceback. `proc.run_logged` converts nonzero exits, missing binaries, and timeouts; everything else raises it deliberately.
3. **Every external effect is an injected callable** (`run`, `which`, `popen`, `urlopen`, `sleep`, `call`, `kill`, `bind`). The test suite (246 tests) runs in under a second with no network, no Docker, and no real processes — and an autouse fixture points `LEPIKA_HOME` at a temp dir so tests can never touch a real `~/.lepika`.
4. **Model refs, one field, three shapes:** `qwen3:8b` (Ollama tag) · `hf.co/<org>/<repo>-GGUF` (Ollama pulls from Hugging Face) · `<org>/<repo>` (full weights → vLLM, Server mode on Linux + NVIDIA; rejected with a GGUF hint anywhere else).
5. **The curated list is data, not code.** `models.toml` ships in the wheel and is re-fetched from `main` at runtime (3s timeout). Remote content is untrusted: unknown keys are dropped, bad types and missing fields skip the entry, parse errors fall back to the bundled copy. The bundled copy parses strictly so defects fail in CI.
6. **Health-check first, act second.** A healthy `lepika up` makes zero subprocess calls and works offline. Restarts wait for the old server to actually die before declaring victory. Server mode is the one exception: `docker compose up -d` is itself the reconciler — idempotent, about a second when nothing changed, and the only thing that makes a `connect` or `expose` edit take effect — so it runs on every `lepika up`.
7. **pid files are hints, not truth.** A pid is only signalled if the webui is actually answering on the configured port — a stale file after a reboot can never kill an unrelated process.

## Notes for maintainers

- **PyPI:** LePika is not published to PyPI yet. Installers therefore install from this repository (`git+https://…`) and must never install a bare name we don't own. Publishing requires registering the distribution name first.
- **Ports:** conflict detection probes both `0.0.0.0` and `127.0.0.1` (macOS lets loopback bind over a wildcard listener) and uses `SO_EXCLUSIVEADDRUSE` on Windows.
- **`ollama` and `open-webui` track latest** by design; `lepika update` is the refresh path. Breakage from upstream is caught by tests + (planned) scheduled smoke runs.
- **OpenWebUI persists its own config:** the first run stores the engine URL from its admin panel and ignores the environment afterwards, so LePika starts it with `ENABLE_PERSISTENT_CONFIG=false` (and `OLLAMA_API_CONFIGS` when the engine needs a key) — otherwise `lepika connect` would move the engine everywhere except in the UI. The environment is read once, at startup, so `lepika connect` also restarts a UI that is already up: `lepika up` alone would leave the healthy old process pointed at the old engine.
- **Known trade-off:** `lepika down` treats a hung-but-alive webui as stale (won't signal it). Follow-up idea: pid liveness fallback.
- **Managed vs. remote engines.** `config.engine_managed` splits them: managed means LePika installs and starts the engine, remote (`lepika connect`) means someone else runs it and LePika may only check that it answers — never install, never start, never stop. `express` exposes one backend surface — `start_stack` / `stop` / `update` / `logs` — that the CLI drives, and `server` mirrors that surface exactly rather than adding per-command branches.
- **Server mode owns its files; you own `.env`.** `server.py` copies `compose.yml`, `compose.nvidia.yml`, and `Caddyfile` out of the wheel into `~/.lepika/stack` on every run, overwriting them — they are LePika's, and local edits are lost. The one file LePika only merges into is `~/.lepika/stack/.env`: it rewrites the keys it manages (ports, engine URL, bind address) and preserves everything else, so pinned image tags and secrets survive. It is written 0600 and atomically, like `config.toml`, because it holds the API key and any Hugging Face token. Compose profiles (`engine`, `vllm`, `expose`) select the optional services; a service whose profile just went inactive is stopped explicitly, because compose will not do it.
- **Exposure is opt-in and always keyed.** `lepika expose` generates `secrets.token_urlsafe(32)` into `.env` *before* starting the stack, then re-runs `start_stack` so the `expose` profile (Caddy) and `WEBUI_BIND=0.0.0.0` apply. `start_stack` refuses the `expose` profile with an empty key: Caddy matches `Bearer {$LEPIKA_API_KEY}`, and an empty key makes a bare `Bearer ` header an open proxy. The key is printed to the terminal only, never logged and never passed in argv.
