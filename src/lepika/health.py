"""Is it answering, and waiting until it is."""

from __future__ import annotations

import urllib.request
from collections.abc import Callable

from lepika.config import Config
from lepika.errors import FriendlyError
from lepika.system import System

_HEALTH_TIMEOUT = 1.0


def api_up(url: str, timeout: float = 1.0, key: str = "", *, sys: System) -> bool:
    request = urllib.request.Request(f"{url}/api/version")
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    try:
        sys.open_url(request, timeout=timeout)
    except Exception:
        return False
    return True


def vllm_up(url: str, *, sys: System) -> bool:
    """Is the vLLM server answering? `GET /health` — a probe, so nothing is logged."""
    try:
        sys.open_url(urllib.request.Request(f"{url}/health"), timeout=_HEALTH_TIMEOUT)
    except Exception:
        return False
    return True


def wait_until(predicate: Callable[[], bool], attempts: int, *, sys: System) -> bool:
    """Check up to `attempts` times, a second apart; did it become true?"""
    for _ in range(attempts):
        if predicate():
            return True
        sys.sleep(1)
    return False


def wait_for(predicate: Callable[[], bool], seconds: int, what: str, *, sys: System) -> None:
    if wait_until(predicate, seconds, sys=sys):
        return
    raise FriendlyError(
        f"{what} did not become ready within {seconds}s.",
        # `lepika logs`, not a raw path: in Server mode the cause is in the
        # container's logs, which no file under ~/.lepika/logs holds.
        "Run `lepika logs` to see why, then `lepika doctor`.",
    )


def webui_url(port: int) -> str:
    return f"http://localhost:{port}"


def webui_up(port: int, *, sys: System) -> bool:
    try:
        sys.open_url(urllib.request.Request(f"http://127.0.0.1:{port}/health"), timeout=1.0)
    except Exception:
        return False
    return True


def engine_unreachable(cfg: Config) -> FriendlyError:
    """What a remote engine that does not answer is, wherever that was found out."""
    return FriendlyError(
        f"The engine at {cfg.engine_url} is not answering.",
        "Make sure that machine is up (and `lepika expose` is on there), or run "
        "`lepika connect --local` to go back to a local engine.",
    )


def check_remote_engine(cfg: Config, *, sys: System) -> None:
    """Confirm an engine we do not manage is answering — the only thing we may do to it."""
    if api_up(cfg.engine_url, key=cfg.engine_key, sys=sys):
        return
    raise engine_unreachable(cfg)
