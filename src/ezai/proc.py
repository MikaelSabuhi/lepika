"""Single choke point for subprocess calls: captured, logged, friendly."""

from __future__ import annotations

import datetime
import subprocess
from collections.abc import Mapping, Sequence

from ezai.errors import FriendlyError
from ezai.paths import logs_dir


def run_logged(
    cmd: Sequence[str],
    *,
    check: bool = True,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        env=dict(env) if env is not None else None,
        timeout=timeout,
    )
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    log_path = logs_dir() / "ezai.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"[{stamp}] $ {' '.join(cmd)}\n"
            f"(exit {result.returncode})\n{result.stdout}{result.stderr}\n"
        )
    if check and result.returncode != 0:
        raise FriendlyError(
            f"Command failed: {' '.join(cmd)}",
            f"Details were logged to {log_path}",
        )
    return result
