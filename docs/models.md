# Model guide

One field, three shapes. The wizard and `lepika model add` both take all of them:

```sh
lepika model add qwen3:8b                          # any tag from the Ollama library
lepika model add hf.co/unsloth/gemma-3-4b-it-GGUF  # any GGUF build on Hugging Face
lepika model add Qwen/Qwen3.8-27B                  # full-weight repo — imported into Ollama, or vLLM in Server mode
```

Run `lepika model add` with no argument and you get the curated list filtered to your
RAM, so there is no guessing whether a 27B fits in 16 GB. Paste a `huggingface.co/…`
link of either kind and LePika works out which it is.

## Choosing the quantization

A GGUF repo usually ships a dozen builds of the same weights at different sizes.
Leave the `:TAG` off and LePika lists them, sized against your machine, with the one
it recommends marked ★ — Enter takes it, a number picks another:

```
      Quantizations of unsloth/Qwen3.8-27B-GGUF
┏━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ # ┃ Quant       ┃    Size ┃ Fit (17 GB GPU)      ┃
┡━━━╇━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ 1 │ UD-IQ2_M    │  9.8 GB │ fits your GPU        │
│ 2 │ UD-Q3_K_XL ★│ 13.1 GB │ fits your GPU        │
│ 3 │ UD-IQ4_XS   │ 14.3 GB │ GPU + some CPU       │
│ 4 │ UD-Q4_K_M   │ 16.5 GB │ GPU + some CPU       │
│ 5 │ UD-Q6_K     │ 22.0 GB │ mostly CPU — slow    │
└───┴─────────────┴─────────┴──────────────────────┘
3 larger builds hidden — they exceed your 32 GB RAM.
Pick a number [2]:
```

"Fits your GPU" leaves a fifth of the GPU's memory for the 16k context window.
The ★ is the largest quantized build that fits there (F16/BF16 are listed, never
recommended). Builds bigger than your RAM are hidden. If the Hub cannot be reached
— offline, or a gated repo without a token — Ollama picks its own default, as before.
Know the tag you want? `hf.co/<org>/<repo>:Q4_K_M` skips the question entirely.
Running without a terminal, say in a script? The ★ is taken for you.

## Full-weight (safetensors) repos

The third shape is the original safetensors release. Where it runs:

| Platform | Support |
| --- | --- |
| macOS (Apple Silicon), Express mode | Out of the box |
| Linux / Windows, Express mode | 64-bit x86 (amd64) NVIDIA GPU with a CUDA 13+ driver (`nvidia-smi` prints it top right) |
| Linux + NVIDIA, Server mode | The same ref runs on vLLM instead |
| Anywhere else | Refused, with a pointer at the GGUF build |

In Express mode LePika downloads the repo and imports it into Ollama with 4-bit
quantization — `nvfp4`, or `--quant int4` if you'd rather have that one. Say yes to the
size it shows first: a 27B is ~55 GB to download and ~15 GB once imported, and the
download is deleted after a successful import. While the import runs you need about
1.3× the download free — ~72 GB for that 27B.

On NVIDIA, the first import also installs Ollama's MLX engine bundle (~1 GB, once, and
a `sudo` prompt on Linux if Ollama lives somewhere you don't own).

## Weights already on disk

```sh
lepika model import ~/models/Qwen3.5-2B
```

imports safetensors weights you already downloaded into Ollama and makes them the
default (`--name` to pick the name, `--quant` to pick the quantization). The folder is
only read, never modified or deleted.

## Gated repos

Gated Hugging Face repos need a token: export `HF_TOKEN` or answer the one-time prompt.
The token is stored in `~/.lepika/config.toml` (mode `0600`); Server mode keeps it in
`stack/.env` (also `0600`). It is never written to logs.
