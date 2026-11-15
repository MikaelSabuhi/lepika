"""Which engine serves a model ref on this machine, in this mode."""

from __future__ import annotations

from lepika.config import Config
from lepika.detect import SystemInfo


def import_allowed(cfg: Config, info: SystemInfo) -> bool:
    """Can this machine import a full-weight repo into Ollama? (rule 10, Express)

    Imports run on Ollama's MLX engine, which is built into the macOS arm64 build.
    On Linux/Windows it is a separate amd64-only CUDA 13 bundle that `ensure_mlx`
    installs on the import path (so an NVIDIA Jetson could never have it, and neither
    could an Intel Mac with an NVIDIA card — there is no macOS bundle at all).
    Express only, and only for an engine that is ours: the weights have to land on
    the engine's machine.
    """
    if not (cfg.mode == "express" and cfg.engine_managed):
        return False
    if info.gpu == "apple":
        return True
    return info.os != "macos" and info.gpu == "nvidia" and info.arch in ("x86_64", "amd64")
