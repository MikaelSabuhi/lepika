<p align="center">
  <img src="https://raw.githubusercontent.com/MikaelSabuhi/lepika/main/docs/assets/logo.png" alt="LePika — a cat in a speech bubble with a terminal prompt" width="220">
</p>

<h1 align="center">LePika</h1>

<p align="center"><b>One command → local AI chat in your browser.</b><br><i>(luh-PEE-ka)</i></p>

<p align="center">
  <a href="https://github.com/MikaelSabuhi/lepika/actions/workflows/ci.yml"><img src="https://github.com/MikaelSabuhi/lepika/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/lepika/"><img src="https://img.shields.io/pypi/v/lepika" alt="PyPI"></a>
  <a href="https://github.com/MikaelSabuhi/lepika/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/MikaelSabuhi/lepika/main/docs/assets/demo.gif" alt="lepika: detect the machine, pick a model, chat UI ready" width="720">
</p>

Your gaming GPU is already an AI machine. LePika self-hosts LLMs on the hardware you
already own — Mac (Metal), Linux and Windows (NVIDIA) — with zero configuration. No
Docker, no API keys, no monthly bill, and nothing you type ever leaves the box.

> Named after Pika, the cat who supervises this project. Like any cat, Pika has
> everything worth having at home and no use for the cloud. Same idea, for your AI.

## Install

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
uv tool install lepika
```

Then run `lepika`. It detects your OS, GPU, and RAM, offers the models that actually
fit your machine, installs [Ollama](https://ollama.com) +
[OpenWebUI](https://openwebui.com), pulls your pick, and opens the chat UI at
`http://localhost:3000`.

> LePika installs into `~/.local/bin` — if a new terminal can't find the
> `lepika` command, add that directory to your `PATH` or restart your shell.

Prefer to read the source first, or hack on it?
`git clone https://github.com/MikaelSabuhi/lepika && cd lepika && uv run lepika` —
git and [uv](https://docs.astral.sh/uv/getting-started/installation/) are the only
prerequisites; uv fetches its own Python.

## What it does

- **Native GPU, no containers** — Apple Metal on macOS, CUDA on Linux/Windows.
  Ollama arrives via Homebrew, the official install script, or winget; OpenWebUI via
  `uv tool`. CPU-only machines work too (LePika warns you they'll be slow).
- **Models that fit** — a curated list filtered to your RAM, so there's no guessing
  whether a 27B fits in 16 GB.
- **Any model, one field** — Ollama tags, Hugging Face GGUF builds, or full-weight
  safetensors repos. [Model guide →](https://github.com/MikaelSabuhi/lepika/blob/main/docs/models.md)
- **Private by default** — everything listens on `127.0.0.1` until you explicitly run
  `lepika expose`.
- **Two modes** — ⚡ **Express** (default): everything native, no Docker.
  🐳 **Server**: the same thing as one readable `docker compose` stack, for the box
  under the desk. [Server guide →](https://github.com/MikaelSabuhi/lepika/blob/main/docs/server-mode.md)

LePika reuses an Ollama you already have instead of installing a second copy.

## Everyday commands

| Command | What it does |
| --- | --- |
| `lepika` | The setup wizard: detect → pick a model → install → chat |
| `lepika up` / `lepika down` | Start / stop the local AI stack |
| `lepika status` | Mode, engine, UI, and default model at a glance |
| `lepika doctor` | Diagnose the setup; every ✗ has a one-line fix |
| `lepika logs` | Tail LePika's logs (`--lines`, default 50) |
| `lepika update` | Upgrade the engine and OpenWebUI — chats kept |
| `lepika model add [ref]` | Download a model and make it the default (no ref → browse) |
| `lepika model import <dir>` | Import safetensors weights already on disk |
| `lepika model list` / `lepika model rm` | List / remove downloaded models |
| `lepika expose` | Share engine + UI on your network behind a generated key |
| `lepika connect <url>` | Use an engine on another machine (`--local` to go back) |

Global flags: `--version`, `--mode express|server` (the wizard's, not a per-command
switch), `--dry-run`, and `--help` on every command.

All state lives in one place: `~/.lepika` — config, logs, and (in Express mode)
OpenWebUI's own chats and uploads. That is what makes `lepika update` keep your chats.
Point `LEPIKA_HOME` somewhere else if you prefer.

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
way. Curious what it's doing on your behalf? It's a few files of plain Python in
[`src/lepika/`](https://github.com/MikaelSabuhi/lepika/tree/main/src/lepika): three dependencies, no magic.

## Requirements

- **Disk** for the models: ~0.5 GB for a tiny one, ~5 GB for a good all-rounder,
  40 GB+ for the flagships.
- **8 GB RAM** recommended. Less works: LePika steers you to smaller models.
- **A GPU is optional.** CPU-only machines run everything, slowly, and LePika says so
  up front.
- **macOS:** [Homebrew](https://brew.sh) (without it, LePika points you at the
  Ollama.app download). **Windows:** winget, which ships with Windows 10/11.
  Full-weight imports have their own fine print — see the [model guide](https://github.com/MikaelSabuhi/lepika/blob/main/docs/models.md).

## Documentation

- [Model guide](https://github.com/MikaelSabuhi/lepika/blob/main/docs/models.md) — every model-ref shape, full-weight imports, gated repos
- [Server mode](https://github.com/MikaelSabuhi/lepika/blob/main/docs/server-mode.md) — the compose stack, using a GPU box remotely, security
- [Architecture](https://github.com/MikaelSabuhi/lepika/blob/main/docs/architecture.md) — design rules and how it's built
- [Changelog](https://github.com/MikaelSabuhi/lepika/blob/main/CHANGELOG.md) — what changed in each release

## Roadmap

Both modes are v0.1 and work today. Next up: **a published package**, so installing is
a plain name instead of a repo URL. Star the repo to follow along, or open an issue
with what you'd want next.

## Contributing

Issues and pull requests are welcome; [CONTRIBUTING.md](https://github.com/MikaelSabuhi/lepika/blob/main/CONTRIBUTING.md) has the short
version of how a change gets in. Found a security problem?
[SECURITY.md](https://github.com/MikaelSabuhi/lepika/blob/main/SECURITY.md) says how to report it privately.

```sh
uv sync --dev && uv run pre-commit install && uv run pytest -q
```

## License

[MIT](https://github.com/MikaelSabuhi/lepika/blob/main/LICENSE) © Mikael Sabuhi
