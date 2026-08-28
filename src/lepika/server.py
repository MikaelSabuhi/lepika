"""Server mode: the docker compose stack. LePika owns the files; you own .env."""

from __future__ import annotations

import importlib.resources
import json
import os
import secrets
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from rich.console import Console
from rich.prompt import Prompt

from lepika import detect, engine, express, log, models, proc
from lepika.config import Config
from lepika.detect import SystemInfo
from lepika.errors import FriendlyError
from lepika.paths import logs_dir, stack_dir

RunFn = Callable[..., Any]
CallFn = Callable[[list[str]], int]
SleepFn = Callable[[float], None]
AskFn = Callable[..., str]

console = Console()

STACK_FILES = ("compose.yml", "compose.nvidia.yml", "Caddyfile")
ENV_FILE = ".env"
VLLM_URL = "http://127.0.0.1:8000"
_TOOLKIT_URL = (
    "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
)
# The weights are tens of GB and vLLM compiles a graph before it answers: minutes,
# sometimes tens of them, on a first start.
_VLLM_READY_SECONDS = 1800

# Written only when absent: these are yours to pin or fill in.
PRESERVED_DEFAULTS = {
    "OPENWEBUI_IMAGE": "ghcr.io/open-webui/open-webui:main",
    "OLLAMA_IMAGE": "ollama/ollama:latest",
    "VLLM_IMAGE": "vllm/vllm-openai:latest",
    "CADDY_IMAGE": "caddy:2",
    # B105: an empty placeholder, not a credential — the real token comes from
    # the shell or from .env.
    "HF_TOKEN": "",  # nosec B105
    "LEPIKA_API_KEY": "",
}
# profile -> the service to stop when that profile is no longer active.
INACTIVE = {"engine": "ollama", "vllm": "vllm", "expose": "caddy"}
# B104: host names compared against an engine URL, never an address we bind.
_LOOPBACK = {"127.0.0.1", "localhost", "0.0.0.0"}  # nosec B104


def install_stack() -> Path:
    """Copy the bundled compose files into ~/.lepika/stack, overwriting: LePika owns them."""
    target = stack_dir()
    bundle = importlib.resources.files("lepika").joinpath("stack")
    for name in STACK_FILES:
        text = bundle.joinpath(name).read_text(encoding="utf-8")
        (target / name).write_text(text, encoding="utf-8")
    return target


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def write_env(path: Path, values: dict[str, str]) -> None:
    """Merge `values` into .env, keep everything else, write privately and atomically."""
    merged = read_env(path) | values
    # Single quotes: compose reads the value literally, so JSON and `$` survive.
    body = "".join(f"{k}='{v}'\n" for k, v in merged.items())
    tmp = path.with_name(path.name + ".tmp")
    # The file holds the API key and an HF token, so it is private from the first
    # byte rather than chmod'ed once the secrets are already on disk. fchmod also
    # tightens a stale tmp file left world-readable by an interrupted run.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    if hasattr(os, "fchmod"):  # POSIX only; Windows ignores mode bits entirely
        os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(body)
    os.replace(tmp, path)
    # Key names only: the values are exactly what must never reach the log.
    log.get_logger().info("env.write", path=str(path), keys=sorted(values))


def api_key(env_path: Path, rotate: bool = False) -> str:
    """The key Caddy checks. Generated once, kept in .env (0600), never in LePika's log."""
    current = read_env(env_path).get("LEPIKA_API_KEY", "")
    if current and not rotate:
        return current
    key = secrets.token_urlsafe(32)
    write_env(env_path, {"LEPIKA_API_KEY": key})
    log.get_logger().info("expose.key", rotated=rotate)
    return key


def lan_ip(connect: Callable[[socket.socket], None] | None = None) -> str:
    """The address other machines reach us on. A UDP connect picks the route; nothing is sent."""
    # 10.255.255.255 needs no DNS and no reachable host: connecting a datagram
    # socket only makes the kernel choose the outbound interface.
    connect_fn = connect if connect is not None else (lambda s: s.connect(("10.255.255.255", 1)))
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            connect_fn(sock)
            return str(sock.getsockname()[0])
    except OSError:
        # No route at all (offline, or a locked-down host): say so in the connect
        # line rather than printing something that looks like an address.
        return "<this machine's IP>"


def container_engine_url(url: str) -> str:
    """A loopback engine URL means the host, which is `host.docker.internal` in a container."""
    parts = urlsplit(url)
    if parts.hostname in _LOOPBACK:
        port = f":{parts.port}" if parts.port else ""
        return urlunsplit((parts.scheme, f"host.docker.internal{port}", parts.path, "", ""))
    return url


def _authority(url: str) -> str:
    """`host[:port]` — what Caddy's `reverse_proxy` wants, with no scheme or path."""
    parts = urlsplit(url)
    return f"{parts.hostname or ''}:{parts.port}" if parts.port else (parts.hostname or "")


def env_values(cfg: Config, info: SystemInfo, existing: dict[str, str]) -> dict[str, str]:
    """Every key compose.yml interpolates — managed ones from config, the rest preserved."""
    values = {k: existing.get(k, v) for k, v in PRESERVED_DEFAULTS.items()}
    # A token in the shell is the freshest one the user has; it wins over the file.
    values["HF_TOKEN"] = os.environ.get("HF_TOKEN", values["HF_TOKEN"])
    engine_url = (
        "http://ollama:11434" if cfg.engine_managed else container_engine_url(cfg.engine_url)
    )
    # Caddy proxies to whatever the engine actually is: with a remote engine the
    # `engine` profile is inactive, so `ollama:11434` would be a 502 on every request.
    upstream = "ollama:11434" if cfg.engine_managed else _authority(engine_url)
    api_configs = json.dumps({"0": {"key": cfg.engine_key}}) if cfg.engine_key else "{}"
    values |= {
        "COMPOSE_PROJECT_NAME": "lepika",
        "WEBUI_PORT": str(cfg.webui_port),
        # B104: reaching the network is exactly what `lepika expose` is for.
        "WEBUI_BIND": "0.0.0.0" if cfg.exposed else "127.0.0.1",  # nosec B104
        "OLLAMA_BASE_URL": engine_url,
        "OLLAMA_API_CONFIGS": api_configs,
        "ENABLE_OPENAI_API": "false",
        "OPENAI_API_BASE_URL": "",
        "OPENAI_API_KEY": "",
        "VLLM_MODEL": "",
        "API_PORT": str(cfg.api_port),
        "LEPIKA_UPSTREAM": upstream,
    }
    if vllm_active(cfg):
        # OLLAMA_BASE_URL is left pointing at the stopped ollama service on purpose:
        # OpenWebUI then lists no Ollama models, so the only model offered is vLLM's.
        values |= {
            "VLLM_MODEL": cfg.model,
            "ENABLE_OPENAI_API": "true",
            "OPENAI_API_BASE_URL": "http://vllm:8000/v1",
            "OPENAI_API_KEY": "none",  # OpenWebUI insists on a value; vLLM ignores it
            "LEPIKA_UPSTREAM": "vllm:8000",
        }
    return values


def vllm_allowed(cfg: Config, info: SystemInfo) -> bool:
    """Could this machine serve a full-weight repo? vLLM is a CUDA container we start.

    `engine_managed` is part of it: with a remote engine we start no containers at
    all, so accepting a repo would save a model nothing in the stack ever serves.
    """
    return (
        cfg.engine_managed and cfg.mode == "server" and info.os == "linux" and info.gpu == "nvidia"
    )


def vllm_active(cfg: Config) -> bool:
    """Is vLLM the engine right now? The one predicate every caller shares."""
    return cfg.engine_managed and models.uses_vllm(cfg.model)


def profiles(cfg: Config) -> list[str]:
    active: list[str] = []
    if cfg.engine_managed:
        active.append("vllm" if vllm_active(cfg) else "engine")
    if cfg.exposed:
        active.append("expose")
    return active


def hf_token_prompt(env_path: Path, ask: AskFn | None = None) -> None:
    """Ask for a Hugging Face token once, only when nothing supplied one. Empty is fine.

    Gated repos (Llama, Gemma…) 401 without one. The answer goes straight to .env
    (0600) — never onto the command line, and never into LePika's log.
    """
    if os.environ.get("HF_TOKEN") or read_env(env_path).get("HF_TOKEN"):
        return
    ask_fn: AskFn = ask if ask is not None else Prompt.ask
    token = ask_fn(
        "Hugging Face token (needed for gated repos; Enter to skip)",
        password=True,
        default="",
    ).strip()
    write_env(env_path, {"HF_TOKEN": token})


def nvidia_in_docker(info: SystemInfo, run: RunFn = proc.run_logged) -> bool:
    """Can Docker hand the NVIDIA GPU to a container? (Linux: the toolkit registers a runtime.)"""
    if info.gpu != "nvidia":
        return False
    if info.os != "linux":
        return True  # Docker Desktop (WSL2) exposes the GPU without a named runtime
    result = run(["docker", "info", "--format", "{{json .Runtimes}}"], check=False, log=False)
    return bool(result.returncode == 0 and "nvidia" in result.stdout)


def compose_cmd(stack: Path, active: list[str], gpu_overlay: bool) -> list[str]:
    cmd = [
        "docker",
        "compose",
        "--project-directory",
        str(stack),
        "-f",
        str(stack / "compose.yml"),
    ]
    if gpu_overlay:
        cmd += ["-f", str(stack / "compose.nvidia.yml")]
    for profile in active:
        cmd += ["--profile", profile]
    return cmd


def ensure_docker(info: SystemInfo, run: RunFn = proc.run_logged) -> None:
    if not info.has_docker:
        raise FriendlyError(
            "Server mode needs Docker, which is not installed.",
            "Install it from https://docs.docker.com/get-docker/ — or use Express mode "
            "(no Docker): `lepika --mode express`.",
        )
    if run(["docker", "info"], check=False, timeout=20, log=False).returncode != 0:
        raise FriendlyError(
            "Docker is installed but not running.",
            "Start Docker Desktop (or `sudo systemctl start docker`) and run `lepika up` again.",
        )


def _compose_failed(step: str) -> FriendlyError:
    return FriendlyError(
        f"`docker compose {step}` failed.",
        "Run `lepika logs` for the container output; `lepika doctor` checks Docker itself.",
    )


def gpu_note(info: SystemInfo, run: RunFn = proc.run_logged) -> str | None:
    """One line when an NVIDIA GPU exists but Docker can't see it; None otherwise."""
    if info.gpu == "nvidia" and info.os == "linux" and not nvidia_in_docker(info, run=run):
        return (
            "NVIDIA GPU found, but Docker can't use it — running on CPU. Install the NVIDIA "
            f"Container Toolkit: {_TOOLKIT_URL}"
        )
    return None


def start_stack(
    info: SystemInfo,
    cfg: Config,
    after_engine: Callable[[], None] | None = None,
    run: RunFn = proc.run_logged,
    call: CallFn = subprocess.call,
    api_up: Callable[..., bool] = detect.api_up,
    vllm_up: Callable[..., bool] = engine.vllm_up,
    up: Callable[..., bool] = express.webui_up,
    sleep: SleepFn = time.sleep,
) -> str:
    """Reconcile the stack with config and return the URL to open.

    Always runs `compose up -d`: it is idempotent, takes about a second when nothing
    changed, and is what makes `connect`/`expose` edits take effect.
    """
    ensure_docker(info, run=run)
    if cfg.engine_managed and info.ollama_running:
        raise FriendlyError(
            "Ollama is already running natively on port 11434, so the Docker engine can't bind it.",
            "Stop it (`brew services stop ollama` / `systemctl stop ollama` / quit it from "
            "the tray), or keep it and run `lepika connect http://127.0.0.1:11434`.",
        )
    stack = install_stack()
    env_path = stack / ENV_FILE
    values = env_values(cfg, info, read_env(env_path))
    active = profiles(cfg)
    gpu = nvidia_in_docker(info, run=run)
    # Both refusals come before anything is written or started.
    # Caddy matches `Authorization: Bearer {$LEPIKA_API_KEY}`; an empty key makes a
    # bare `Bearer ` header match, which is an open proxy on the network.
    if "expose" in active and not values["LEPIKA_API_KEY"]:
        raise FriendlyError(
            "Network exposure is on, but there is no API key to protect it.",
            "Run `lepika expose` to generate one, or `lepika expose --off`.",
        )
    # vLLM has no CPU fallback worth offering: without the GPU the container starts
    # and then dies, which is a far worse way to learn the toolkit is missing.
    if "vllm" in active and not gpu:
        raise FriendlyError(
            "vLLM needs the GPU inside Docker, and Docker can't see it.",
            f"Install the NVIDIA Container Toolkit: {_TOOLKIT_URL}",
        )
    write_env(env_path, values)
    base = compose_cmd(stack, active, gpu_overlay=gpu)
    log.get_logger().info("stack.up", profiles=active, gpu=info.gpu)
    # Streamed, not captured: the first run pulls gigabytes of images and a silent
    # terminal for minutes looks like a hang.
    if call([*base, "up", "-d", "--remove-orphans"]) != 0:
        raise _compose_failed("up")
    # Compose does not stop services whose profile just went inactive; we do.
    for profile, service in INACTIVE.items():
        if profile not in active:
            run(
                [*compose_cmd(stack, [profile], gpu_overlay=False), "stop", service],
                check=False,
                timeout=60,
            )
    if "vllm" in active:
        # The first start downloads the full weights — tens of GB. A silent wait
        # that long is indistinguishable from a hang, so say what is happening.
        console.print(
            "vLLM is loading the model — the first time this downloads the full "
            "weights; watch with `lepika logs`."
        )
        express.wait_for(lambda: vllm_up(VLLM_URL), _VLLM_READY_SECONDS, "vLLM", sleep=sleep)
    elif cfg.engine_managed:
        express.wait_for(lambda: api_up(cfg.engine_url), 120, "Ollama (container)", sleep=sleep)
    else:
        express.check_remote_engine(cfg, api_up=api_up)
    if after_engine is not None:
        after_engine()
    express.wait_for(lambda: up(cfg.webui_port), 180, "OpenWebUI", sleep=sleep)
    return express.webui_url(cfg.webui_port)


def stop(info: SystemInfo, cfg: Config, run: RunFn = proc.run_logged) -> bool:
    """`lepika down` in Server mode: stop and remove the containers; volumes stay."""
    ensure_docker(info, run=run)
    stack = install_stack()
    every = list(INACTIVE)  # every profile, so nothing is left behind
    result = run([*compose_cmd(stack, every, gpu_overlay=False), "down"], check=False, timeout=120)
    log.get_logger().info("stack.down", exit=result.returncode)
    return bool(result.returncode == 0)


def update(
    info: SystemInfo,
    cfg: Config,
    run: RunFn = proc.run_logged,
    call: CallFn = subprocess.call,
) -> None:
    """`lepika update` in Server mode: pull newer images, then reconcile."""
    ensure_docker(info, run=run)
    stack = install_stack()
    log.get_logger().info("stack.update")
    if call([*compose_cmd(stack, profiles(cfg), gpu_overlay=False), "pull"]) != 0:
        raise _compose_failed("pull")
    start_stack(info, cfg, run=run, call=call)


def logs(lines: int, run: RunFn = proc.run_logged) -> list[tuple[str, str]]:
    """`lepika logs` in Server mode: compose logs, then LePika's own log."""
    stack = stack_dir()
    result = run(
        [
            *compose_cmd(stack, list(INACTIVE), gpu_overlay=False),
            "logs",
            "--no-color",
            "--tail",
            str(lines),
        ],
        check=False,
        timeout=30,
        log=False,
    )
    compose_output = result.stdout if result.returncode == 0 else "(docker compose is not running)"
    sections = [("docker compose", compose_output)]
    lepika_log = logs_dir() / log.LOG_FILE
    if lepika_log.exists():
        text = lepika_log.read_text(encoding="utf-8", errors="replace").splitlines()
        sections.append((log.LOG_FILE, "\n".join(text[-lines:])))
    return sections
