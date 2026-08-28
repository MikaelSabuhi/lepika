# LePika

**One command → local AI chat in your browser.**

*(luh-PEE-ka)*

Your gaming GPU is already an AI machine. LePika self-hosts LLMs on the hardware you
already own, with zero configuration: Mac (Metal), Linux and Windows (NVIDIA).
No Docker, no API keys, no monthly bill, and nothing you type ever leaves the box.

> A pika is a small mountain mammal that spends the summer stashing hay in its own
> burrow, so everything it needs is already at home. Same idea, for your AI.

[![CI](https://github.com/MikaelSabuhi/lepika/actions/workflows/ci.yml/badge.svg)](https://github.com/MikaelSabuhi/lepika/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

<!-- demo GIF placeholder: added by the release plan via vhs -->

## Install

> **v0.1 is landing.** The one-liners below work today — they install straight
> from this repository. A published package is planned; until then, installing by
> name could fetch someone else's project, so LePika never does.

**Mac / Linux**

```sh
curl -fsSL https://raw.githubusercontent.com/MikaelSabuhi/lepika/main/install.sh | sh
```

**Windows (PowerShell)**

```powershell
irm https://raw.githubusercontent.com/MikaelSabuhi/lepika/main/install.ps1 | iex
```

**Already have [uv](https://docs.astral.sh/uv/)?**

```sh
uv tool install git+https://github.com/MikaelSabuhi/lepika
```

A published package will shorten this to a plain name once one exists.

LePika installs into `~/.local/bin`. If a new terminal can't find the `lepika` command,
add that directory to your `PATH` or restart your shell.

That's it. LePika detects your OS, GPU, and RAM, offers the models that actually fit
your machine, installs [Ollama](https://ollama.com) + [OpenWebUI](https://openwebui.com),
pulls your pick, and opens the chat UI at `http://localhost:3000`.

### From a clone

Prefer to read the source first, or hack on it:

```sh
git clone https://github.com/MikaelSabuhi/lepika
cd lepika
uv run lepika
```

Only prerequisites are git and
[uv](https://docs.astral.sh/uv/getting-started/installation/). uv fetches its own
Python, so there is nothing else to set up.

## What you get

Everything runs natively on your machine. No containers, no VMs, no cloud.

| Platform | GPU used | How LePika sets it up |
| --- | --- | --- |
| macOS (Apple Silicon) | Apple Metal | Ollama via Homebrew, OpenWebUI via `uv tool` |
| macOS (Intel) | CPU | Same, minus the GPU |
| Linux + NVIDIA | CUDA | Official Ollama install script, OpenWebUI via `uv tool` |
| Linux (no GPU) | CPU | Same; LePika warns you it will be slow |
| Windows + NVIDIA | CUDA | `winget install Ollama.Ollama`, OpenWebUI via `uv tool` |
| Windows (no GPU) | CPU | Same; LePika warns you it will be slow |

Ollama already installed? LePika reuses it instead of installing a second copy.

## Everyday commands

| Command | What it does |
| --- | --- |
| `lepika` | The setup wizard: detect → pick a model → install → open the browser |
| `lepika up` | Start the local AI stack and open the browser |
| `lepika down` | Stop OpenWebUI (Ollama keeps running as a shared service) |
| `lepika status` | Show what's running |
| `lepika logs` | Print the tail of LePika's log files (`--lines`, default 50) |
| `lepika doctor` | Diagnose the local setup; every ✗ comes with a one-line fix |
| `lepika update` | Upgrade Ollama and OpenWebUI to their latest versions |
| `lepika model add [ref]` | Download a model and make it the default (no ref → browse) |
| `lepika model list` | List downloaded models (size, default marked) |
| `lepika model rm <name>` | Remove a downloaded model |

Global flags: `--version`, `--dry-run` (show what the wizard would do without doing
it), and `--help` on every command.

All state lives in one place: `~/.lepika` (config, pid files, logs). Point `LEPIKA_HOME`
somewhere else if you prefer.

## Pick any model

One field, three shapes. The wizard and `lepika model add` both take all of them:

```sh
lepika model add qwen3:8b                          # any tag from the Ollama library
lepika model add hf.co/unsloth/gemma-3-4b-it-GGUF  # any GGUF build on Hugging Face
lepika model add meta-llama/Llama-3.3-70B-Instruct # full-weight HF repo — needs vLLM
```

The third shape needs vLLM, which arrives with Server mode; today LePika says so and
points you at the GGUF build of the same model.

Run `lepika model add` with no argument and you get the curated list filtered to your
RAM, so there is no guessing whether a 27B fits in 16 GB.

## Requirements

- **Disk** for the models: ~0.5 GB for a tiny one, ~5 GB for a good all-rounder,
  40 GB+ for the flagships.
- **8 GB RAM** recommended. Less works: LePika just steers you to smaller models.
- **A GPU is optional.** CPU-only machines run everything; they run it slowly, and
  LePika tells you so up front.
- **macOS:** [Homebrew](https://brew.sh) for the Ollama install (without it, LePika
  points you at the Ollama.app download). **Windows:** winget, which ships with
  Windows 10/11.

## Why not just use Ollama directly?

You absolutely can. LePika drives [Ollama](https://ollama.com) and
[OpenWebUI](https://openwebui.com); it doesn't replace them. What it saves you:

| | By hand | With LePika |
| --- | --- | --- |
| First run | Install Ollama, install Python, install open-webui, set `OLLAMA_BASE_URL`, start both, find the port | One command |
| Choosing a model | Guess, download 20 GB, find out it swaps | A list filtered to the RAM you actually have |
| Wiring UI ↔ engine | Environment variables, by hand, per shell | Done for you |
| Something's broken | Two sets of logs, two projects' issue trackers | `lepika doctor`, one fix hint per failure |
| Staying current | Update each piece separately, per platform | `lepika update` |

If you'd rather run the raw tools, their docs are excellent and LePika gets out of your
way. And if you're curious what it's doing on your behalf, it's under a thousand lines
of plain Python in [`src/lepika/`](src/lepika): three dependencies, no magic.

## Roadmap

Express mode (above) is v0.1 and works today. Next up:

- **🐳 Server mode** — a single, readable `docker compose` stack: OpenWebUI + Ollama,
  with a **vLLM** profile for full-weight Hugging Face repos on Linux + NVIDIA.
- **`lepika expose`** — serve the API to your network behind a generated API key
  (Caddy Bearer auth), instead of localhost only.
- **`lepika connect <url>`** — point the UI at an engine running on another box.
- **A published package** — so installing is a plain name instead of a repo URL.
  The install one-liners already work today either way.

Star the repo to follow along, or open an issue with what you'd want next.

## Contributing

Work happens in a git worktree on a feature branch and lands via PR; `main` is never
committed to directly. Conventional commits (`feat:`, `fix:`, `docs:`…), squash-merged.

```sh
uv sync --dev
uv run pre-commit install
uv run pytest -q
```

Every PR runs the same gate; run it locally in one line before you push:

```sh
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q && uv run bandit -c pyproject.toml -r src -q && uv run pip-audit
```

## License

[MIT](LICENSE) © Mikael Sabuhi
