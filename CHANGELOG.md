# Changelog

All notable changes to LePika are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
LePika follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Every model now gets a 16k-token context window. Ollama's own default is 4096,
  so one pasted document or a chat an hour old failed with "request exceeds the
  available context size" on any model. LePika sets `OLLAMA_CONTEXT_LENGTH` on the
  `ollama serve` it starts in Express mode and on the engine container in Server
  mode; a new `context_length` key in `config.toml` changes it, and a value already
  in the shell wins over the file. When Ollama was already running — the tray app
  on Windows, brew, systemd — `lepika up` says where that engine's own setting is.
- `lepika model add hf.co/<org>/<repo>-GGUF` (and the same ref typed in the wizard)
  now shows the repo's quantizations sized against your GPU and RAM, marks the one
  that fits best ★, and takes it on Enter — no more hand-typing `:UD-Q4_K_M` after
  reading file sizes on Hugging Face. An explicit `:TAG` behaves exactly as before,
  and so does a pull when the Hub cannot be reached. ([#5])

### Fixed

- Changing `context_length` in `config.toml` and running `lepika up` now takes
  effect. `lepika down` leaves the Ollama LePika started running, so `up` used to
  adopt it with the old value and say nothing; it now restarts its own engine when
  the setting changed. An engine started by LePika 0.1.x, which never set a context
  length and so runs at Ollama's 4096, is restarted the same way. `lepika model add`
  and `lepika model import` start the engine with the configured value too, instead
  of always 16384.
- In Server mode on macOS and Windows the quantization picker and the model list
  now size models against the memory Docker Desktop gives its VM, not the
  machine's. On a 16 GB Mac that VM has under 8 GB, so the ★ the picker
  recommended could never load in the engine container. When Docker's limit is
  what hides a build, the picker says where to raise it.
- The quantization picker shows an NVIDIA card's memory in the same GB as RAM, so
  a 16 GB card reads as 16 GB rather than 17.

## [0.1.2] - 2026-08-31

### Fixed

- A bundled ChatML template missing from a damaged install now raises a
  `FriendlyError` instead of a bare `FileNotFoundError`. The template repair runs
  after a pull has already succeeded, so the traceback would have been the only
  thing the user saw go wrong. ([#15])
- `lepika down` now stops an OpenWebUI that `lepika up` adopted rather than
  started. `up` returns early when the port already answers, recording no pid, and
  `down` had only that pid file to go on — so it reported "Nothing was running"
  while the UI kept serving and `lepika status` kept showing it up. It now falls
  back to the process list, the same command-line evidence a hung pid is already
  judged by. ([#17])
- `lepika update` no longer reports "Ollama installation failed" after the engine
  upgraded fine. The official install script runs under `set -e`, so a machine
  whose systemd is wedged aborts it in the optional service setup — the very setup
  LePika undoes on the next line — long after the binary landed. The engine on
  PATH now decides, and the unit the aborted script left enabled is disabled as
  usual. ([#18])

## [0.1.1] - 2026-08-31

### Fixed

- Repos whose weights are already quantized (NVFP4, AWQ and friends) are imported
  as-is, instead of failing after the full multi-gigabyte download because Ollama
  refuses to requantize a quantized source. The size estimate no longer promises a
  store copy a quarter the download's size. ([#10])
- The systemd `ollama.service` that Ollama's install script enables on Linux is
  disabled after install, so it stops fighting LePika's own `ollama serve` for port
  11434 — and stops winning that port after a reboot with an empty model store. ([#7])
- A failed Express-to-Server switch now sweeps up the containers it left behind.
  Previously they stayed invisible to `lepika down` while holding ports 3000 and
  11434. ([#8])
- Pulled ChatML models whose published template dropped its tool handling are
  rebuilt with a tool-capable chat template. ([#12])
- `quant_method`, read from a repo's own `config.json`, is escaped before it is
  printed. A value containing Rich markup was parsed as markup and silently vanished
  from the text you read to approve a download. ([#13])

## [0.1.0] - 2026-08-30

Initial public release — one command → local AI chat in your browser.

- Two modes: ⚡ Express (default, no Docker, native GPU) and 🐳 Server (Docker
  Compose), with a guided setup wizard.
- Commands: `up`, `down`, `status`, `logs`, `doctor`, `update`, `connect`, `expose`,
  and `model` for adding, listing and removing local models.
- A curated in-repo model catalogue, plus importing safetensors weights from
  Hugging Face or a local folder.

[Unreleased]: https://github.com/MikaelSabuhi/lepika/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/MikaelSabuhi/lepika/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/MikaelSabuhi/lepika/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MikaelSabuhi/lepika/releases/tag/v0.1.0
[#5]: https://github.com/MikaelSabuhi/lepika/issues/5
[#7]: https://github.com/MikaelSabuhi/lepika/pull/7
[#8]: https://github.com/MikaelSabuhi/lepika/pull/8
[#10]: https://github.com/MikaelSabuhi/lepika/pull/10
[#12]: https://github.com/MikaelSabuhi/lepika/pull/12
[#13]: https://github.com/MikaelSabuhi/lepika/pull/13
[#15]: https://github.com/MikaelSabuhi/lepika/pull/15
[#17]: https://github.com/MikaelSabuhi/lepika/pull/17
[#18]: https://github.com/MikaelSabuhi/lepika/pull/18
