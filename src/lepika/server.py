"""Server mode: the docker compose stack. LePika owns the files; you own .env."""

from __future__ import annotations

import importlib.resources
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from lepika import health, log, serving, stack_env
from lepika.config import Config
from lepika.detect import Machine
from lepika.errors import FriendlyError
from lepika.paths import logs_dir, stack_dir
from lepika.system import System
from lepika.ui import UI
from lepika.updates import UpdateStep

STACK_FILES = ("compose.yml", "compose.nvidia.yml", "Caddyfile")
VLLM_URL = "http://127.0.0.1:8000"
# Where compose.yml publishes the engine container on this machine. Not
# `config.DEFAULT_ENGINE_URL`: that is where a managed engine is reached, and
# `engine_url` is the user's to edit; this port is the stack's own.
HOST_ENGINE_URL = "http://127.0.0.1:11434"
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


def lan_ip(*, sys: System) -> str:
    """The address other machines reach us on: the one the default route leaves from."""
    route = sys.route_address()
    if route is None:
        # No route at all (offline, or a locked-down host): say so in the connect
        # line rather than printing something that looks like an address.
        return NO_ROUTE
    return route


def lan_ips(*, sys: System) -> list[str]:
    """Every address other machines could reach us on, the default route's pick first.

    One address is the usual answer, but a box on Wi-Fi and Ethernet (or behind a
    VPN, or with a Docker bridge) answers on several, and the route pick is only
    the one the kernel would use outbound — not necessarily the one the laptop in
    the next room can reach.
    """
    found: list[str] = []
    route = lan_ip(sys=sys)
    if route != NO_ROUTE:
        found.append(route)
    # An unresolvable hostname is normal on a laptop; the route pick still stands.
    for ip in sys.host_addresses():
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


def env_values(cfg: Config, existing: dict[str, str], environ: Mapping[str, str]) -> dict[str, str]:
    """Every key compose.yml interpolates — managed ones from config, the rest preserved."""
    values = {k: existing.get(k, v) for k, v in PRESERVED_DEFAULTS.items()}
    # A token in the shell is the freshest one the user has; it wins over the file.
    values["HF_TOKEN"] = environ.get("HF_TOKEN", values["HF_TOKEN"])
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
    if serving.vllm_active(cfg):
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


def profiles(cfg: Config) -> list[str]:
    active: list[str] = []
    if cfg.engine_managed:
        active.append("vllm" if serving.vllm_active(cfg) else "engine")
    if cfg.exposed:
        active.append("expose")
    return active


def hf_token_prompt(env_path: Path, *, sys: System, ui: UI) -> None:
    """Ask for a Hugging Face token once, only when nothing supplied one. Empty is fine.

    Gated repos (Llama, Gemma…) 401 without one. The answer goes straight to .env
    (0600) — never onto the command line, and never into LePika's log.
    """
    if sys.environ.get("HF_TOKEN") or stack_env.read_env(env_path).get("HF_TOKEN"):
        return
    # No terminal skips, as Enter does.
    token = ui.ask(
        "Hugging Face token (needed for gated repos; Enter to skip)",
        password=True,
        default="",
    ).strip()
    stack_env.write_env(env_path, {"HF_TOKEN": token})


def nvidia_in_docker(info: Machine, *, sys: System) -> bool:
    """Can Docker hand the NVIDIA GPU to a container? (Linux: the toolkit registers a runtime.)"""
    if info.gpu != "nvidia":
        return False
    if info.os != "linux":
        return True  # Docker Desktop (WSL2) exposes the GPU without a named runtime
    result = sys.run(
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


def ensure_docker(retry: str = "lepika up", *, sys: System) -> None:
    if sys.which("docker") is None:
        raise FriendlyError(
            "Server mode needs Docker, which is not installed.",
            "Install it from https://docs.docker.com/get-docker/ — or use Express mode "
            "(no Docker): `lepika --mode express`.",
        )
    if sys.run(["docker", "info"], check=False, timeout=20, log=False).returncode != 0:
        raise FriendlyError(
            "Docker is installed but not running.",
            f"Start Docker Desktop (or `sudo systemctl start docker`) and run `{retry}` again.",
        )


def _our_ollama(stack: Path, *, sys: System) -> bool:
    """Is the engine answering on 11434 our own container rather than a native Ollama?

    The stack publishes exactly `127.0.0.1:11434`, so every run after the first sees
    the port taken — by itself. Compose is the only thing that can tell them apart.
    """
    result = sys.run(
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


def gpu_note(info: Machine, *, sys: System) -> str | None:
    """One line when an NVIDIA GPU exists but Docker can't see it; None otherwise."""
    if info.gpu == "nvidia" and info.os == "linux" and not nvidia_in_docker(info, sys=sys):
        return (
            "NVIDIA GPU found, but Docker can't use it — running on CPU. Install the NVIDIA "
            f"Container Toolkit: {_TOOLKIT_URL}"
        )
    return None


def start_stack(
    info: Machine,
    cfg: Config,
    after_engine: Callable[[], None] | None = None,
    *,
    sys: System,
    ui: UI,
) -> str:
    """Reconcile the stack with config and return the URL to open.

    Always runs `compose up -d`: it is idempotent, takes about a second when nothing
    changed, and is what makes `connect`/`expose` edits take effect.
    """
    ensure_docker(sys=sys)
    stack = install_stack()
    # Asked now, not taken from detection: the wizard may have just stopped the Ollama
    # that held the port. Cheapest first; no key, as it is a managed engine's port.
    if (
        cfg.engine_managed
        and health.api_up(HOST_ENGINE_URL, sys=sys)
        and not _our_ollama(stack, sys=sys)
    ):
        raise FriendlyError(
            "Ollama is already running natively on port 11434, so the Docker engine can't bind it.",
            "Stop it first: `brew services stop ollama` (macOS), `systemctl stop ollama` "
            "(Linux), or quit it from the tray (Windows); if LePika started it, "
            '`pkill -f "ollama serve"`.',
        )
    env_path = stack / stack_env.ENV_FILE
    values = env_values(cfg, stack_env.read_env(env_path), sys.environ)
    active = profiles(cfg)
    gpu = nvidia_in_docker(info, sys=sys)
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
    stack_env.write_env(env_path, values)
    base = compose_cmd(stack, active, gpu_overlay=gpu)
    log.get_logger().info("stack.up", mode="server", profiles=active, gpu=info.gpu)
    # Streamed, not captured: the first run pulls gigabytes of images and a silent
    # terminal for minutes looks like a hang.
    if sys.call([*base, "up", "-d", "--remove-orphans"]) != 0:
        raise _compose_failed("up")
    # Compose does not stop services whose profile just went inactive; we do —
    # in one call, since compose takes several profiles and several services.
    inactive = [profile for profile in INACTIVE if profile not in active]
    if inactive:
        sys.run(
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
        ui.say(
            "vLLM is loading the model — the first time this downloads the full "
            "weights; watch with `lepika logs`."
        )
        health.wait_for(
            lambda: health.vllm_up(VLLM_URL, sys=sys), _VLLM_READY_SECONDS, "vLLM", sys=sys
        )
    elif cfg.engine_managed:
        health.wait_for(
            lambda: health.api_up(cfg.engine_url, sys=sys), 120, "Ollama (container)", sys=sys
        )
    else:
        health.check_remote_engine(cfg, sys=sys)
    if after_engine is not None:
        after_engine()
    health.wait_for(lambda: health.webui_up(cfg.webui_port, sys=sys), 180, "OpenWebUI", sys=sys)
    return health.webui_url(cfg.webui_port)


def stop(info: Machine, cfg: Config, *, sys: System) -> bool:
    """`lepika down` in Server mode: stop and remove the containers; volumes stay."""
    ensure_docker(retry="lepika down", sys=sys)
    stack = install_stack()
    every = list(INACTIVE)  # every profile, so nothing is left behind
    result = sys.run(
        [*compose_cmd(stack, every, gpu_overlay=False), "down"], check=False, timeout=120
    )
    log.get_logger().info("stack.down", exit=result.returncode)
    return bool(result.returncode == 0)


def update(
    info: Machine,
    cfg: Config,
    *,
    sys: System,
    ui: UI,
) -> list[UpdateStep]:
    """`lepika update` in Server mode: pull newer images, then reconcile."""
    ensure_docker(retry="lepika update", sys=sys)
    stack = install_stack()
    log.get_logger().info("stack.update")
    base = compose_cmd(stack, profiles(cfg), gpu_overlay=False)
    before = _image_ids(base, sys=sys)
    if sys.call([*base, "pull"]) != 0:
        raise _compose_failed("pull")
    after = _image_ids(base, sys=sys)
    start_stack(info, cfg, sys=sys, ui=ui)
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


def _image_ids(base: list[str], *, sys: System) -> str:
    """The local image ids behind the stack's images: what a pull that changed nothing keeps.

    Compared as a whole rather than parsed: an image not pulled yet makes `inspect`
    fail on that name, and that too changes once the pull has fetched it. Empty when
    there is nothing to compare — no images listed, or no id read for any of them.
    """
    listed = sys.run([*base, "config", "--images"], check=False, log=False)
    images = str(listed.stdout or "").split()
    if not images:
        return ""
    ids = sys.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", *images], check=False, log=False
    )
    if not str(ids.stdout or "").strip():
        return ""
    return f"{ids.returncode}\n{ids.stdout}"


def logs(lines: int, *, sys: System) -> list[tuple[str, str]]:
    """`lepika logs` in Server mode: compose logs, then LePika's own log."""
    stack = stack_dir()
    result = sys.run(
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
