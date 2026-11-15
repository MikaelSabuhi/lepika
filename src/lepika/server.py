"""Server mode: the docker compose stack. LePika owns the files; you own .env."""

from __future__ import annotations

import functools
import importlib.resources
import json
import os
import secrets
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from rich.console import Console

from lepika import fsio, health, log, models, proc, prompts, system
from lepika.backend import UpdateStep
from lepika.config import Config
from lepika.detect import SystemInfo
from lepika.errors import FriendlyError
from lepika.paths import logs_dir, stack_dir

RunFn = Callable[..., Any]
CallFn = Callable[[list[str]], int]
SleepFn = Callable[[float], None]
AskFn = Callable[..., str]

# The default probes, bound once at import as the bare functions used to be: a test
# that patches `health.api_up` or `health.webui_up` does not reach a default.
_REAL_API_UP = functools.partial(health.api_up, sys=system.REAL)
_REAL_WEBUI_UP = functools.partial(health.webui_up, sys=system.REAL)

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
# Printed in place of an address when the machine has no route out at all.
NO_ROUTE = "<this machine's IP>"


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
            quote, value = value[0], value[1:-1]
            if quote == '"':
                value = _unescape(value)
        values[key.strip()] = value
    return values


def _unescape(text: str) -> str:
    """Undo `_quote`'s double-quoted escapes: `\\"`, `\\\\`, and `$$` — nothing else."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] in '\\"':
            out.append(text[i + 1])
            i += 2
        elif text[i] == "$" and i + 1 < len(text) and text[i + 1] == "$":
            out.append("$")
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _quote(value: str) -> str:
    """Single quotes keep JSON and `$` literal — but they cannot hold a `'`.

    A value containing one (an API key from `lepika connect --key`) is written
    double-quoted instead, with the escapes compose's .env parser expects — `$`
    included, because compose interpolates `$VAR` inside double quotes.
    """
    if "'" not in value:
        return f"'{value}'"
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "$$")
    return f'"{escaped}"'


def write_env(path: Path, values: dict[str, str]) -> None:
    """Merge `values` into .env, keep everything else, write privately and atomically."""
    merged = read_env(path) | values
    body = "".join(f"{k}={_quote(v)}\n" for k, v in merged.items())
    # Private, because the file holds the API key and an HF token.
    fsio.write_private(path, body)
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
        return NO_ROUTE


def lan_ips(
    connect: Callable[[socket.socket], None] | None = None,
    getaddrinfo: Callable[..., Any] = socket.getaddrinfo,
    hostname: Callable[[], str] = socket.gethostname,
) -> list[str]:
    """Every address other machines could reach us on, the default route's pick first.

    One address is the usual answer, but a box on Wi-Fi and Ethernet (or behind a
    VPN, or with a Docker bridge) answers on several, and the route pick is only
    the one the kernel would use outbound — not necessarily the one the laptop in
    the next room can reach.
    """
    found: list[str] = []
    route = lan_ip(connect=connect)
    if route != NO_ROUTE:
        found.append(route)
    try:
        addresses = getaddrinfo(hostname(), None, socket.AF_INET)
    except OSError:
        # An unresolvable hostname is normal on a laptop; the route pick still stands.
        addresses = []
    for entry in addresses:
        ip = str(entry[4][0])
        # Loopback and link-local are addresses nobody on the network dials.
        if ip.startswith(("127.", "169.254.")) or ip in found:
            continue
        found.append(ip)
    return found


def container_engine_url(url: str) -> str:
    """A loopback engine URL means the host, which is `host.docker.internal` in a container."""
    parts = urlsplit(url)
    if parts.hostname in _LOOPBACK:
        port = f":{parts.port}" if parts.port else ""
        return urlunsplit((parts.scheme, f"host.docker.internal{port}", parts.path, "", ""))
    return url


def _upstream(url: str) -> str:
    """`LEPIKA_UPSTREAM` for Caddy's `reverse_proxy`: `host[:port]`, no path.

    A bare `host:port` makes Caddy speak plain HTTP to the upstream, so an engine
    behind TLS keeps its `https://` — that scheme is the only way to ask for it.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    authority = f"{host}:{parts.port}" if parts.port else host
    return f"https://{authority}" if parts.scheme == "https" else authority


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
    upstream = "ollama:11434" if cfg.engine_managed else _upstream(engine_url)
    api_configs = json.dumps({"0": {"key": cfg.engine_key}}) if cfg.engine_key else "{}"
    values |= {
        "COMPOSE_PROJECT_NAME": "lepika",
        "WEBUI_PORT": str(cfg.webui_port),
        # B104: reaching the network is exactly what `lepika expose` is for.
        "WEBUI_BIND": "0.0.0.0" if cfg.exposed else "127.0.0.1",  # nosec B104
        "OLLAMA_BASE_URL": engine_url,
        "OLLAMA_API_CONFIGS": api_configs,
        # The engine container's default context; compose recreates it on a change.
        "OLLAMA_CONTEXT_LENGTH": str(cfg.context_length),
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
    # No terminal skips, as Enter does.
    token = prompts.ask(
        "Hugging Face token (needed for gated repos; Enter to skip)",
        password=True,
        default="",
        reader=ask,
    ).strip()
    write_env(env_path, {"HF_TOKEN": token})


def nvidia_in_docker(info: SystemInfo, run: RunFn = proc.run_logged) -> bool:
    """Can Docker hand the NVIDIA GPU to a container? (Linux: the toolkit registers a runtime.)"""
    if info.gpu != "nvidia":
        return False
    if info.os != "linux":
        return True  # Docker Desktop (WSL2) exposes the GPU without a named runtime
    result = run(
        ["docker", "info", "--format", "{{json .Runtimes}}"], check=False, timeout=20, log=False
    )
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


def ensure_docker(info: SystemInfo, run: RunFn = proc.run_logged, retry: str = "lepika up") -> None:
    if not info.has_docker:
        raise FriendlyError(
            "Server mode needs Docker, which is not installed.",
            "Install it from https://docs.docker.com/get-docker/ — or use Express mode "
            "(no Docker): `lepika --mode express`.",
        )
    if run(["docker", "info"], check=False, timeout=20, log=False).returncode != 0:
        raise FriendlyError(
            "Docker is installed but not running.",
            f"Start Docker Desktop (or `sudo systemctl start docker`) and run `{retry}` again.",
        )


def _our_ollama(stack: Path, run: RunFn) -> bool:
    """Is the engine answering on 11434 our own container rather than a native Ollama?

    The stack publishes exactly `127.0.0.1:11434`, so every run after the first sees
    the port taken — by itself. Compose is the only thing that can tell them apart.
    """
    result = run(
        [
            *compose_cmd(stack, ["engine"], gpu_overlay=False),
            "ps",
            "-q",
            "--status",
            "running",
            "ollama",
        ],
        check=False,
        log=False,
        timeout=30,
    )
    return bool(result.returncode == 0 and result.stdout.strip())


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


def _real_vllm_up(url: str) -> bool:
    return health.vllm_up(url, sys=system.REAL)


def start_stack(
    info: SystemInfo,
    cfg: Config,
    after_engine: Callable[[], None] | None = None,
    run: RunFn = proc.run_logged,
    call: CallFn = proc.call,
    api_up: Callable[..., bool] = _REAL_API_UP,
    vllm_up: Callable[..., bool] | None = None,
    up: Callable[..., bool] = _REAL_WEBUI_UP,
    sleep: SleepFn = time.sleep,
) -> str:
    """Reconcile the stack with config and return the URL to open.

    Always runs `compose up -d`: it is idempotent, takes about a second when nothing
    changed, and is what makes `connect`/`expose` edits take effect.
    """
    vllm_probe = vllm_up if vllm_up is not None else _real_vllm_up
    waits = system.overlay(system.REAL, sleep=sleep)
    ensure_docker(info, run=run)
    stack = install_stack()
    if cfg.engine_managed and info.ollama_running and not _our_ollama(stack, run):
        raise FriendlyError(
            "Ollama is already running natively on port 11434, so the Docker engine can't bind it.",
            "Stop it first: `brew services stop ollama` (macOS), `systemctl stop ollama` "
            "(Linux), or quit it from the tray (Windows); if LePika started it, "
            '`pkill -f "ollama serve"`.',
        )
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
    log.get_logger().info("stack.up", mode="server", profiles=active, gpu=info.gpu)
    # Streamed, not captured: the first run pulls gigabytes of images and a silent
    # terminal for minutes looks like a hang.
    if call([*base, "up", "-d", "--remove-orphans"]) != 0:
        raise _compose_failed("up")
    # Compose does not stop services whose profile just went inactive; we do —
    # in one call, since compose takes several profiles and several services.
    inactive = [profile for profile in INACTIVE if profile not in active]
    if inactive:
        run(
            [
                *compose_cmd(stack, inactive, gpu_overlay=False),
                "stop",
                *(INACTIVE[profile] for profile in inactive),
            ],
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
        health.wait_for(lambda: vllm_probe(VLLM_URL), _VLLM_READY_SECONDS, "vLLM", sys=waits)
    elif cfg.engine_managed:
        health.wait_for(lambda: api_up(cfg.engine_url), 120, "Ollama (container)", sys=waits)
    elif not api_up(cfg.engine_url, key=cfg.engine_key):
        # The injected probe stands in for `health.check_remote_engine`'s own.
        raise health.engine_unreachable(cfg)
    if after_engine is not None:
        after_engine()
    health.wait_for(lambda: up(cfg.webui_port), 180, "OpenWebUI", sys=waits)
    return health.webui_url(cfg.webui_port)


def stop(info: SystemInfo, cfg: Config, run: RunFn = proc.run_logged) -> bool:
    """`lepika down` in Server mode: stop and remove the containers; volumes stay."""
    ensure_docker(info, run=run, retry="lepika down")
    stack = install_stack()
    every = list(INACTIVE)  # every profile, so nothing is left behind
    result = run([*compose_cmd(stack, every, gpu_overlay=False), "down"], check=False, timeout=120)
    log.get_logger().info("stack.down", exit=result.returncode)
    return bool(result.returncode == 0)


def update(
    info: SystemInfo,
    cfg: Config,
    run: RunFn = proc.run_logged,
    call: CallFn = proc.call,
) -> list[UpdateStep]:
    """`lepika update` in Server mode: pull newer images, then reconcile."""
    ensure_docker(info, run=run, retry="lepika update")
    stack = install_stack()
    log.get_logger().info("stack.update")
    base = compose_cmd(stack, profiles(cfg), gpu_overlay=False)
    before = _image_ids(base, run)
    if call([*base, "pull"]) != 0:
        raise _compose_failed("pull")
    after = _image_ids(base, run)
    start_stack(info, cfg, run=run, call=call)
    if not after:
        # Nothing to compare: "up to date" would be a claim nobody checked. Unreadable
        # before but readable after is different: the pull certainly fetched something.
        return [
            UpdateStep(
                "Container images",
                "note",
                "pulled the latest images; could not tell whether anything changed",
            )
        ]
    return [UpdateStep("Container images", "current" if before == after else "upgraded")]


def _image_ids(base: list[str], run: RunFn) -> str:
    """The local image ids behind the stack's images: what a pull that changed nothing keeps.

    Compared as a whole rather than parsed: an image not pulled yet makes `inspect`
    fail on that name, and that too changes once the pull has fetched it. Empty when
    there is nothing to compare — no images listed, or no id read for any of them.
    """
    listed = run([*base, "config", "--images"], check=False, log=False)
    images = str(listed.stdout or "").split()
    if not images:
        return ""
    ids = run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", *images], check=False, log=False
    )
    if not str(ids.stdout or "").strip():
        return ""
    return f"{ids.returncode}\n{ids.stdout}"


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
