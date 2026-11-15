"""Is it answering, and waiting until it is."""

from __future__ import annotations

import time
import urllib.request
from collections.abc import Callable
from typing import Any

from lepika.config import Config
from lepika.errors import FriendlyError
from lepika.system import System

UrlOpenFn = Callable[..., Any]
SleepFn = Callable[[float], None]

_HEALTH_TIMEOUT = 1.0


def api_up(
    url: str,
    timeout: float = 1.0,
    urlopen: UrlOpenFn | None = None,
    key: str = "",
) -> bool:
    opener: UrlOpenFn = urlopen if urlopen is not None else urllib.request.urlopen
    request = urllib.request.Request(f"{url}/api/version")
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    try:
        opener(request, timeout=timeout)
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


def wait_for(
    predicate: Callable[[], bool],
    seconds: int,
    what: str,
    sleep: SleepFn = time.sleep,
) -> None:
    for _ in range(seconds):
        if predicate():
            return
        sleep(1)
    raise FriendlyError(
        f"{what} did not become ready within {seconds}s.",
        # `lepika logs`, not a raw path: in Server mode the cause is in the
        # container's logs, which no file under ~/.lepika/logs holds.
        "Run `lepika logs` to see why, then `lepika doctor`.",
    )


def webui_url(port: int) -> str:
    return f"http://localhost:{port}"


def webui_up(port: int, urlopen: Callable[..., Any] | None = None) -> bool:
    opener: Callable[..., Any] = urlopen if urlopen is not None else urllib.request.urlopen
    try:
        opener(f"http://127.0.0.1:{port}/health", timeout=1.0)
    except Exception:
        return False
    return True


def check_remote_engine(cfg: Config, api_up: Callable[..., bool] = api_up) -> None:
    """Confirm an engine we do not manage is answering — the only thing we may do to it."""
    if api_up(cfg.engine_url, key=cfg.engine_key):
        return
    raise FriendlyError(
        f"The engine at {cfg.engine_url} is not answering.",
        "Make sure that machine is up (and `lepika expose` is on there), or run "
        "`lepika connect --local` to go back to a local engine.",
    )
