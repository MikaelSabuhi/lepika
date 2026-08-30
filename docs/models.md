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
