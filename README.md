# LePika

**One command → local AI chat in your browser.**

*(luh-PEE-ka)*

Your gaming GPU is already an AI machine. LePika self-hosts LLMs on the hardware you
already own, with zero configuration: Mac (Metal), Linux and Windows (NVIDIA).
No Docker, no API keys, no monthly bill, and nothing you type ever leaves the box.
Got Docker and a homelab? Server mode runs the same thing as one `docker compose` stack.

> A pika is a small mountain mammal that spends the summer stashing hay in its own
> burrow, so everything it needs is already at home. Same idea, for your AI.

[![CI](https://github.com/MikaelSabuhi/lepika/actions/workflows/ci.yml/badge.svg)](https://github.com/MikaelSabuhi/lepika/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

<!-- demo GIF goes here — see the Roadmap -->

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

### ⚡ Express mode (default)

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

### 🐳 Server mode (`lepika --mode server`)

| Platform | GPU used | Stack |
| --- | --- | --- |
| Linux + NVIDIA | CUDA (needs NVIDIA Container Toolkit) | OpenWebUI + Ollama containers, + vLLM for full-weight repos |
| Linux (no GPU) | CPU | Same; LePika warns you it will be slow |
| Windows + NVIDIA (Docker Desktop) | CUDA via WSL2 | Same |
| macOS (Docker Desktop) | CPU only — containers can't use Metal; use Express for the GPU | Same |

Server mode is asked about only when Docker is already installed; LePika never asks you
to install Docker.

## Everyday commands

| Command | What it does |
| --- | --- |
| `lepika` | The setup wizard: detect → (Express or Server) → pick a model → install → open the browser |
| `lepika up` | Start the local AI stack and open the browser |
| `lepika down` | Stop the stack (Express: OpenWebUI only; Server: all containers, models kept) |
| `lepika status` | Show the mode, the engine and where it is, OpenWebUI, and the default model |
| `lepika logs` | Print the tail of LePika's logs (Server mode: container logs too; `--lines`, default 50) |
| `lepika doctor` | Diagnose the local setup; every ✗ comes with a one-line fix |
| `lepika update` | Upgrade the engine and OpenWebUI (Server mode: pull newer images, then restart) |
| `lepika connect <url> [--key K]` | Use an engine on another machine (`--local` to go back) |
| `lepika expose [--off\|--show\|--rotate]` | Share the engine + UI on your network behind a generated API key (Server mode) |
| `lepika model add [ref]` | Download a model and make it the default (no ref → browse) |
| `lepika model list` | List downloaded models (size, default marked) |
| `lepika model rm <name>` | Remove a downloaded model |

Global flags: `--version`, `--mode express|server` (the wizard's, not a per-command
switch), `--dry-run` (show what the wizard would do without doing it), and `--help` on
every command.

All state lives in one place: `~/.lepika` — config, logs, pid files, and in Server mode the
compose stack. Point `LEPIKA_HOME` somewhere else if you prefer.

## Server mode

Everything in one readable `docker compose` file, for the box under the desk:

```sh
lepika --mode server            # or pick 🐳 in the wizard when Docker is present
```

The stack lives in `~/.lepika/stack/`. LePika owns `compose.yml`; you own `.env`
(created private, `0600`). Pin a version by editing it — `OLLAMA_IMAGE='ollama/ollama:0.11.4'` —
and LePika keeps your pins on every `lepika up` / `lepika update`. Models and chats live in named
Docker volumes and survive `lepika down`.

Both modes serve the UI on the same port, so switching between them stops the stack
you're leaving before it starts the one you're moving to.

**Security.** Nothing listens beyond localhost until you run `lepika expose`. Then the
chat UI needs a sign-in (first sign-up is the admin) and the engine API needs the
generated key. The key lives only in `~/.lepika/stack/.env` (mode `0600`) on the box and
in `~/.lepika/config.toml` (`0600`) on machines you connected from; it is never written
to logs.

## Pick any model

One field, three shapes. The wizard and `lepika model add` both take all of them:

```sh
lepika model add qwen3:8b                          # any tag from the Ollama library
lepika model add hf.co/unsloth/gemma-3-4b-it-GGUF  # any GGUF build on Hugging Face
lepika model add meta-llama/Llama-3.3-70B-Instruct # full-weight HF repo — vLLM, Server mode on Linux + NVIDIA
```

The third shape runs on vLLM inside the Server-mode stack (Linux + NVIDIA only, and only
with LePika's own engine — not a `connect`ed one; anywhere else LePika refuses it and
points you at the GGUF build). Gated repos need a Hugging Face token: export `HF_TOKEN`
or answer the one-time prompt; it is stored in `~/.lepika/stack/.env` (0600).

Run `lepika model add` with no argument and you get the curated list filtered to your
RAM, so there is no guessing whether a 27B fits in 16 GB.

## Use a GPU box from your laptop

The chat UI runs where you are; the models run where the GPU is.

```sh
lepika expose                                     # on the box: prints the key and the exact connect line
lepika connect http://gpu-box:11435 --key <key>   # on your laptop: paste that line
lepika up                                         # UI here, models there
lepika connect --local                            # back to this machine
```

`lepika expose` (Server mode) puts a small Caddy proxy in front of the engine on port
`11435`: only requests carrying the generated key get through. `--show` reprints the
line, `--rotate` issues a new key, `--off` goes back to localhost only. If the box runs a
full-weight repo on vLLM, `lepika expose` prints an OpenAI-compatible URL to add in
OpenWebUI instead of a `connect` line.

LePika never installs or starts anything on an engine it didn't set up — it only
checks that it answers, and says so plainly when it doesn't. If the chat UI is
already running, `lepika connect` restarts it so the switch takes effect right away.

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
way. And if you're curious what it's doing on your behalf, it's a few files of plain
Python in [`src/lepika/`](src/lepika): three dependencies, no magic.

## Roadmap

Both modes above are v0.1 and work today. Still to come:

- **A published package** — so installing is a plain name instead of a repo URL.
  The install one-liners already work today either way.
- **A scheduled smoke run**, so an upstream change to Ollama or OpenWebUI breaks CI
  before it breaks you.
- **A demo GIF** at the top of this page, because one command deserves one picture.

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
