# Changelog

All notable changes to LePika are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
LePika follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- A bundled ChatML template missing from a damaged install now raises a
  `FriendlyError` instead of a bare `FileNotFoundError`. The template repair runs
  after a pull has already succeeded, so the traceback would have been the only
  thing the user saw go wrong. ([#15])

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

[Unreleased]: https://github.com/MikaelSabuhi/lepika/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/MikaelSabuhi/lepika/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MikaelSabuhi/lepika/releases/tag/v0.1.0
[#7]: https://github.com/MikaelSabuhi/lepika/pull/7
[#8]: https://github.com/MikaelSabuhi/lepika/pull/8
[#10]: https://github.com/MikaelSabuhi/lepika/pull/10
[#12]: https://github.com/MikaelSabuhi/lepika/pull/12
[#13]: https://github.com/MikaelSabuhi/lepika/pull/13
[#15]: https://github.com/MikaelSabuhi/lepika/pull/15
