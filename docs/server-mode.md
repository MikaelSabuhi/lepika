# Server mode

Everything in one readable `docker compose` file, for the box under the desk:

```sh
lepika --mode server            # or pick 🐳 in the wizard when Docker is present
```

Server mode is asked about only when Docker is already installed; LePika never asks
you to install Docker.

| Platform | GPU used | Stack |
| --- | --- | --- |
| Linux + NVIDIA | CUDA (needs NVIDIA Container Toolkit) | OpenWebUI + Ollama containers, + vLLM for full-weight repos |
| Linux (no GPU) | CPU | Same; LePika warns you it will be slow |
| Windows + NVIDIA (Docker Desktop) | CUDA via WSL2 | Same |
| macOS (Docker Desktop) | CPU only — containers can't use Metal; use Express for the GPU | Same |

## The stack

The stack lives in `~/.lepika/stack/`. LePika owns `compose.yml`; you own `.env`
(created private, `0600`). Pin a version by editing it —
`OLLAMA_IMAGE='ollama/ollama:0.11.4'` — and LePika keeps your pins on every
`lepika up` / `lepika update`. Models and chats live in named Docker volumes and
survive `lepika down`.

Both modes serve the UI on the same port, so switching between them stops the stack
you're leaving before it starts the one you're moving to — including the native Ollama
LePika started for Express, since the Server stack wants that port. An Ollama you
installed and run yourself is never stopped.

## Security

Nothing LePika starts listens beyond localhost in either mode — Express starts
OpenWebUI on `127.0.0.1`, and the Server stack publishes every port there too — until
you run `lepika expose`. Then the chat UI needs a sign-in (first sign-up is the admin)
and the engine API needs the generated key. The key lives only in
`~/.lepika/stack/.env` (mode `0600`) on the box and in `~/.lepika/config.toml`
(`0600`) on machines you connected from; it is never written to logs. Found a security
problem? [SECURITY.md](../SECURITY.md) says how to report it privately.

## Use a GPU box from your laptop

The chat UI runs where you are; the models run where the GPU is.

```sh
lepika expose                                     # on the box: prints the key and the exact connect line
lepika connect http://gpu-box:11435 --key <key>   # on your laptop: paste that line
lepika up                                         # UI here, models there
lepika connect --local                            # back to this machine
```

`lepika expose` (Server mode) puts a small Caddy proxy in front of the engine on port
`11435`: only requests carrying the generated key get through. It shares the engine
this machine runs — or an unkeyed remote one — and refuses an engine that needs its
own key, since the proxy would forward LePika's key to it and be told no every time
(`lepika connect --key` refuses on an exposed machine for the same reason). It prints
the address this box would dial out on, plus every other address it answers on when
there is more than one — the one your laptop can reach is not always the first.

`--show` reprints the line, `--rotate` issues a new key (machines that already
connected have to run `lepika connect` again with it), `--off` goes back to localhost
only. If the box runs a full-weight repo on vLLM, `lepika expose` prints an
OpenAI-compatible URL to add in OpenWebUI instead of a `connect` line.

LePika never installs or starts anything on an engine it didn't set up — it only
checks that it answers, and says so plainly when it doesn't. If the chat UI is already
running, `lepika connect` restarts it so the switch takes effect right away.
