"""Which engine serves a model ref on this machine, in this mode."""

from __future__ import annotations

from lepika import models
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


def vllm_allowed(cfg: Config, info: SystemInfo) -> bool:
    """Could this machine serve a full-weight repo? vLLM is a CUDA container we start.

    `engine_managed` is part of it: with a remote engine we start no containers at
    all, so accepting a repo would save a model nothing in the stack ever serves.
    """
    return (
        cfg.engine_managed and cfg.mode == "server" and info.os == "linux" and info.gpu == "nvidia"
    )


def vllm_active(cfg: Config) -> bool:
    """Is vLLM the engine right now? The one predicate every caller shares.

    The mode is part of the answer: vLLM is a container in the Server stack, so an
    Express config still naming a repo (a switch that failed halfway) has no vLLM
    behind it for `status`, `doctor`, or `model list` to report.
    """
    return cfg.mode == "server" and cfg.engine_managed and models.uses_vllm(cfg.model)


def engine_label(cfg: Config) -> str:
    """What to call the engine in a sentence: the model picks it (rule 10).

    A remote engine is nameless on purpose — `lepika connect` never asks what runs
    over there, so calling it "Ollama" would be a guess printed as a fact.
    """
    if vllm_active(cfg):
        return "vLLM"
    return "Ollama" if cfg.engine_managed else "a remote engine"
