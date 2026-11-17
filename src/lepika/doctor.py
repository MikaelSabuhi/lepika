"""lepika doctor: every red ✗ comes with a one-line fix."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from lepika import config, health, server, serving
from lepika.detect import Machine
from lepika.errors import FriendlyError
from lepika.system import System

# Advisory, not a hard failure: `cli.doctor` matches on this name, not a literal.
RAM_CHECK = "RAM"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    hint: str = ""


def _finding(probe: Callable[[], bool]) -> bool:
    """Run a probe and turn its refusal into a ✗ row rather than an abort.

    `run_logged` raises on a timeout or a missing binary; from a diagnosis both are
    findings — a Docker daemon mid-start accepts the socket and then says nothing,
    and every probe here is bounded so that it says so instead of hanging.
    """
    try:
        return probe()
    except FriendlyError:
        return False


def _answers(cmd: list[str], *, sys: System) -> bool:
    """Does `cmd` exit 0 — and does it answer at all?"""
    return _finding(lambda: bool(sys.run(cmd, check=False, timeout=20, log=False).returncode == 0))


def run_checks(info: Machine, *, sys: System) -> list[CheckResult]:
    cfg = config.load()
    checks: list[CheckResult] = []
    if cfg.mode == "server":
        # Server mode installs nothing natively, so uv and a local Ollama are
        # irrelevant; Docker is the whole prerequisite. `and` short-circuits, so a
        # machine with no docker binary is never probed with one.
        docker_ok = bool(sys.which("docker") is not None and _answers(["docker", "info"], sys=sys))
        checks += [
            CheckResult(
                "Docker running",
                docker_ok,
                "Install/start Docker: https://docs.docker.com/get-docker/ — "
                "or `lepika --mode express`",
            ),
            CheckResult(
                "docker compose available",
                bool(docker_ok and _answers(["docker", "compose", "version"], sys=sys)),
                "Docker Compose v2 ships with Docker Desktop; on Linux: docker-compose-plugin",
            ),
        ]
        # Gated on docker_ok: probing the GPU through a Docker that is not there
        # would raise instead of reporting, and Docker is the one root cause anyway.
        if docker_ok and info.gpu == "nvidia" and info.os == "linux":
            checks.append(
                CheckResult(
                    "NVIDIA GPU visible to Docker",
                    _finding(lambda: server.nvidia_in_docker(info, sys=sys)),
                    "Install the NVIDIA Container Toolkit: https://docs.nvidia.com/"
                    "datacenter/cloud-native/container-toolkit/install-guide.html",
                )
            )
    else:
        checks.append(
            CheckResult(
                "uv installed",
                sys.which("uv") is not None,
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
    vllm = serving.vllm_active(cfg)
    checks += [
        CheckResult(
            "Engine (vLLM) responding" if vllm else "Engine responding",
            health.vllm_up(server.VLLM_URL, sys=sys)
            if vllm
            else health.api_up(cfg.engine_url, key=cfg.engine_key, sys=sys),
            "Run `lepika up` to start it; logs: `lepika logs`"
            if cfg.engine_managed
            # A rotated key and a dead box look identical from here, so name both.
            else f"{cfg.engine_url} is not answering — check that machine, or the key "
            f"changed: `lepika connect {cfg.engine_url} --key <key>` — or "
            "`lepika connect --local`",
        ),
        CheckResult(
            "OpenWebUI responding",
            health.webui_up(cfg.webui_port, sys=sys),
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
