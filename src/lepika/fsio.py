"""Writing files that hold secrets: privately, atomically, from the first byte."""

from __future__ import annotations

import os
from pathlib import Path


def write_private(path: Path, text: str) -> None:
    """Atomically replace `path` with `text`, mode 0600 from the first byte.

    The mode is set when the temp file is created, not chmod'ed once the secret is
    already on disk under the process umask, and `fchmod` also tightens a stale
    temp file an interrupted run left world-readable, which `O_TRUNC` alone would
    keep. The rename means a write that dies halfway leaves the old file intact.
    """
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    if hasattr(os, "fchmod"):  # POSIX only; Windows ignores mode bits entirely
        os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)
