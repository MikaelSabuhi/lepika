# Security

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's private
reporting instead: **Security → Report a vulnerability** on this repository. You will
get an acknowledgement within a few days and a fix or a clear answer as soon as the
problem is understood.

## What counts

LePika is a thin layer that installs and starts Ollama, OpenWebUI, and a
`docker compose` stack. Report to LePika when the problem is in that layer — for
example:

- a generated API key or Hugging Face token reaching a log, the command line, or a
  file that is not `0600`;
- `lepika expose` letting a request through without the key, or something listening
  beyond localhost without `expose`;
- the installers (`install.sh`, `install.ps1`) fetching or running anything other than
  uv and this package;
- a curated `models.toml` entry that could make LePika run something unexpected.

A problem inside a model, Ollama, OpenWebUI, vLLM, or Caddy belongs with that project;
LePika drives them and does not patch them.

## Supported versions

Only the latest release receives fixes. `lepika update` keeps the engine and the chat UI
current; upgrade LePika itself with `uv tool upgrade lepika`.
