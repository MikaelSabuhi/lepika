# ezai — Design Spec

**Date:** 2026-08-27
**Status:** Approved design, pre-implementation
**Repo:** ezaiselfhost

## 1. Product

One command → local AI chat in your browser. `ezai` is a cross-platform
CLI (Mac, Linux, Windows) that sets up a locally hosted LLM served
through OpenWebUI, using the GPU the machine actually has (Apple
Metal/MPS, NVIDIA CUDA) with zero manual configuration.

**Primary persona:** anyone with a capable machine (gaming PC, MacBook,
homelab box) and no dev-tools expertise. **Secondary persona:** homelab /
self-host enthusiasts who want a compose stack, vLLM throughput, and
network exposure.

**Success criteria:**
- Fresh Windows gaming PC or Mac: paste one line, answer the wizard
  (≤4 questions), chatting in the browser in under ~2 minutes plus
  model download time.
- Repo is viral-ready: `uvx ezai` / per-OS one-liner above the fold,
  GIF demo, transparent readable source.

**Non-goals (v1):** non-LLM models (image/audio), SSH provisioning of
remote machines, Kubernetes, auto-updating (Watchtower), Ollama
library/HF live search UI.

## 2. Two modes, one wizard

| | ⚡ Express (default) | 🐳 Server |
|---|---|---|
| Engine | Native Ollama (installer per OS) | Ollama container; optional vLLM profile |
| UI | OpenWebUI via `uv tool` (pip package) | OpenWebUI container (`ghcr.io/open-webui/open-webui:main`) |
| GPU | Native on all platforms (Metal, CUDA) | CUDA passthrough (Linux/Windows); macOS uses hybrid: native Ollama + containerized UI |
| Requires | Nothing (installer bootstraps uv) | Docker Engine / Docker Desktop |
| Extras | — | vLLM profile, Caddy API-key exposure |

Mode selection logic in the wizard:
- Docker not installed → Express, silently. Never ask a non-Docker user
  to install Docker.
- Docker present → default Express, offer Server.
- User wants vLLM, network API exposure, or an always-on server →
  Server (CLI explains why).
- Remote connect mode (no local engine) works identically in both.

## 3. Platform matrix

| Detected | Express strategy | Server strategy |
|---|---|---|
| macOS (Apple Silicon / Intel) | Ollama via Homebrew (or Ollama.app download link if no brew); OpenWebUI via uv | Native Ollama (Metal) + OpenWebUI container → `host.docker.internal:11434` |
| Linux + NVIDIA | Ollama official install script (systemd service); OpenWebUI via uv | Ollama container `gpus: all`, or vLLM profile |
| Linux CPU-only | Same, with explicit "CPU = slow" warning | Ollama container (CPU) |
| Windows + NVIDIA | `winget install Ollama.Ollama`; OpenWebUI via uv | Ollama container via Docker Desktop/WSL2 |
| Remote | No engine; OpenWebUI → remote URL + optional Bearer key | Same |

Detection inputs: `platform`, `uname -m`, `nvidia-smi` presence/output,
total RAM (and VRAM when detectable), Docker presence + daemon state,
existing Ollama installation (reuse it — never double-install).

## 4. Architecture

```
ezai CLI (Python 3.11+, Typer + Rich, uv-managed)
 ├─ detect.py   platform / GPU / RAM / docker / existing installs
 ├─ wizard.py   interactive flow (≤4 questions)
 ├─ models.py   model-ref parsing, curated list, RAM-fit filter
 ├─ express.py  native installs, open-webui process mgmt (pid file)
 ├─ stack.py    compose orchestration: .env generation, profiles
 ├─ remote.py   connect-only remote endpoints
 └─ doctor.py   diagnostics + guided fixes
State: ~/.ezai/{config.toml, stack/, logs/}
```

**Server-mode compose** (`stack/docker-compose.yml`, single file,
profiles):
- `openwebui` — always on
- `ollama` — profile `engine`
- `vllm` — profile `vllm` (`vllm/vllm-openai:latest`, Linux+NVIDIA)
- `caddy` — profile `expose` (Bearer-token reverse proxy in front of
  the engine API)

All image tags are `.env` variables with `:main`/`:latest` defaults —
users can pin in one line when upstream breaks. Engine API ports bind
to `127.0.0.1` unless `expose` is enabled.

**Config contract:** `config.toml` carries `schema_version`; future
versions migrate rather than fail. CLI commands, env schema, and
`~/.ezai` layout are the public API (semver-major on break).

## 5. CLI surface

```
ezai                  # wizard: detect → ask → install → up → open browser
ezai up / down / status / logs [service]
ezai model add [ref] / list / rm
ezai connect <url>    # remote engine (prompts for optional API key)
ezai expose           # Server mode only: caddy profile + generated API key
ezai update           # newest images / native upgrades, restart
ezai doctor           # diagnostics with guided fixes
```

Wizard: (1) announce detected plan in one sentence; (2) local or
remote; (3) model pick — curated list filtered to RAM/VRAM fit, or
free-form; (4) [Server mode] expose to network? Then up, pull with
progress bar, open browser.

**Model refs — one field, three shapes:**
- `qwen3:8b` → Ollama library tag (Ollama pulls newest by tag)
- `hf.co/<org>/<repo>-GGUF[:quant]` → Ollama pulls GGUF from HF
- `<org>/<repo>` (HF repo id) → vLLM profile (Linux+NVIDIA Server mode
  only; elsewhere CLI explains and suggests the GGUF route). Prompts
  for `HF_TOKEN` on gated repos.

**Curated list:** `models.toml` in-repo, shipped in the wheel; at
runtime the CLI attempts to fetch the newest copy from the repo's main
branch (3 s timeout, silent fallback). No scraping, no live-API
dependency.

## 6. Authentication

- Browser: OpenWebUI built-in accounts; first signup becomes admin.
- API/network: `ezai expose` (Server mode) generates an API key, prints
  it once, and fronts the engine API with Caddy Bearer-token auth.
  Express mode v1 does not expose the API; the CLI directs users to
  Server mode for that.
- `ezai connect` supports remote endpoints with or without a key.

## 7. Error handling

Every failure maps to a one-line next step; no stack traces.
- Docker missing/stopped → per-OS instructions; macOS offers
  `open -a Docker`.
- GPU invisible in container → `doctor` runs `nvidia-smi` in a
  throwaway container; distinguishes driver vs container-toolkit vs
  Docker Desktop WSL setting.
- Port conflicts → name the squatter, offer alternate port into `.env`.
- Model larger than RAM/VRAM → warn before download, suggest largest
  fitting variant.
- All subprocess output logged to `~/.ezai/logs/` (doctor + bug
  reports cite it).

## 8. Testing (TDD throughout)

1. **Unit** (no Docker): detection with faked `uname`/`nvidia-smi`/
   winget outputs; model-ref parsing; `.env` generation; RAM-fit
   filtering; config migration. The bulk of tests.
2. **Integration** (CI Linux): `docker compose config` across all
   profile combinations; wizard end-to-end in `--dry-run` (writes
   files, executes nothing) for both modes.
3. **Smoke** (weekly + on release): real CPU Server stack → pull
   `qwen3:0.6b` → one chat completion through OpenWebUI API → down.
   Catches upstream `:latest`/`:main` breakage.

## 9. Release engineering & maintainability

- **Principles:** thin orchestration (never reimplement upstream);
  two runtime deps (Typer, Rich); zero live scraping; escape hatches
  over pins.
- **Development workflow — PR-only, worktree-based:** `main` is
  protected; no direct pushes. Every change happens in a git worktree
  on a feature branch and lands via a PR that must pass CI. Small,
  focused PRs (one plan task or fix each). PRs are **squash-and-merged**
  into `main` on approval, giving one clean conventional commit per
  change.
- **Styling:** `ruff format` (Black-style) is the single formatter,
  enforced locally via **pre-commit hooks** (`.pre-commit-config.yaml`
  running ruff-check + ruff-format) and verified in CI. No debates, no
  drift.
- **CI pipeline (one workflow, uv-native, fast):** every PR runs, via
  `astral-sh/setup-uv` with caching:
  1. `ruff check` + `ruff format --check` (lint/format)
  2. `mypy` in strict mode (types)
  3. `pytest` unit + integration (`uv run pytest`)
  4. Security audit: `bandit` (code) + `pip-audit` (dependency CVEs)
  That's the whole gate — no sprawling job matrix; smoke tests stay on
  the weekly schedule + release tags, not on every PR.
- **Releases — semver, tag-driven:** tag `vX.Y.Z` on `main` → full CI
  + smoke test → PyPI via trusted publishing (OIDC, no secrets) →
  GitHub Release with generated notes. Version lives in one place
  (`pyproject.toml`).
- **Hygiene:** conventional commits → CHANGELOG; Dependabot (Python
  deps + Actions); issue template requires `ezai doctor` output;
  CONTRIBUTING.md documents the superpowers workflow (brainstorm →
  spec → plan → TDD, worktrees, PR-only) as the project's development
  process.

## 10. Distribution & virality kit

- **Install lines (above the fold in README):**
  - Windows: `irm https://ezai.sh/install.ps1 | iex`
  - Mac/Linux: `curl -fsSL https://ezai.sh/install | sh`
  - Already have uv: `uvx ezai`
  (Installer scripts live in-repo; the `ezai.sh` domain is aspirational —
  raw GitHub URLs work day one.)
- **Repo layout:** `stack/` (compose, Caddyfile, .env.example — visible
  and hackable), `src/ezai/`, `install.sh` + `install.ps1`,
  `models.toml`, `pyproject.toml`.
- **README:** vhs-recorded GIF (regenerable), platform matrix,
  "why not just X?" honesty table, MIT license. Launch demo video for
  HN/X/r/LocalLLaMA. The README is a first-class deliverable: every
  user-facing change (commands, install lines, platform support) must
  update it in the same PR — it stays current, clean, and attractive.
- **Repo hygiene:** clean root (only files users need to see), proper
  `.gitignore`, CONTRIBUTING.md, MIT LICENSE, issue templates.
