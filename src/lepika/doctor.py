"""lepika doctor: every red ✗ comes with a one-line fix."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from lepika import config, detect, engine, express, proc, server
from lepika.detect import SystemInfo

# Advisory, not a hard failure: `cli.doctor` matches on this name, not a literal.
RAM_CHECK = "RAM"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    hint: str = ""


def run_checks(
    info: SystemInfo,
    which: Callable[[str], str | None] = shutil.which,
    api_up: Callable[..., bool] = detect.api_up,
    vllm_up: Callable[..., bool] = engine.vllm_up,
    webui_up: Callable[..., bool] = express.webui_up,
    run: Callable[..., Any] = proc.run_logged,
) -> list[CheckResult]:
    cfg = config.load()
    checks: list[CheckResult] = []
    if cfg.mode == "server":
        # Server mode installs nothing natively, so uv and a local Ollama are
        # irrelevant; Docker is the whole prerequisite. `and` short-circuits, so a
        # machine with no docker binary is never probed with one.
        docker_ok = bool(
            which("docker") is not None
            and run(["docker", "info"], check=False, timeout=20, log=False).returncode == 0
        )
        checks += [
            CheckResult(
                "Docker running",
                docker_ok,
                "Install/start Docker: https://docs.docker.com/get-docker/ — "
                "or `lepika --mode express`",
            ),
            CheckResult(
                "docker compose available",
                bool(
                    docker_ok
                    and run(["docker", "compose", "version"], check=False, log=False).returncode
                    == 0
                ),
                "Docker Compose v2 ships with Docker Desktop; on Linux: docker-compose-plugin",
            ),
        ]
        # Gated on docker_ok: probing the GPU through a Docker that is not there
        # would raise instead of reporting, and Docker is the one root cause anyway.
        if docker_ok and info.gpu == "nvidia" and info.os == "linux":
            checks.append(
                CheckResult(
                    "NVIDIA GPU visible to Docker",
                    server.nvidia_in_docker(info, run=run),
                    "Install the NVIDIA Container Toolkit: https://docs.nvidia.com/"
                    "datacenter/cloud-native/container-toolkit/install-guide.html",
                )
            )
    else:
        checks.append(
            CheckResult(
                "uv installed",
                which("uv") is not None,
                "Install uv: https://docs.astral.sh/uv/getting-started/installation/",
            )
        )
        if cfg.engine_managed:
            # Only asked when the engine is ours to install: a remote one is not.
            checks.append(
                CheckResult(
                    "Ollama installed",
                    info.has_ollama,
                    "Run `lepika` to install it, or see https://ollama.com/download",
                )
            )
    vllm = server.vllm_active(cfg)
    checks += [
        CheckResult(
            "Engine (vLLM) responding" if vllm else "Engine responding",
            vllm_up(server.VLLM_URL) if vllm else api_up(cfg.engine_url, key=cfg.engine_key),
            "Run `lepika up` to start it; logs: `lepika logs`"
            if cfg.engine_managed
            # A rotated key and a dead box look identical from here, so name both.
            else f"{cfg.engine_url} is not answering — check that machine, or the key "
            f"changed: `lepika connect {cfg.engine_url} --key <key>` — or "
            "`lepika connect --local`",
        ),
        CheckResult(
            "OpenWebUI responding",
            webui_up(cfg.webui_port),
            "Run `lepika up`; if it fails, `lepika logs` shows openwebui.log; "
            f"if the port is busy, change webui_port in {config.config_path()}",
        ),
        CheckResult(
            RAM_CHECK,
            info.ram_gb >= 8.0,
            f"{info.ram_gb:.0f} GB detected — 8 GB+ recommended; stick to small "
            "models like qwen3:0.6b or llama3.2:3b",
        ),
    ]
    return checks
