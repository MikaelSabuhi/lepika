"""One JSON-lines log for everything LePika does, with secrets redacted."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import structlog

from lepika.paths import logs_dir

LOG_FILE = "lepika.log"

# Any event key that looks like a credential is masked before it is rendered.
# The log is what users are asked to paste into issues; it must be safe to share.
_SECRET_KEYS = frozenset({"key", "api_key", "token", "hf_token", "engine_key", "password"})


def _redact(
    _logger: Any, _method: str, event: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for name in _SECRET_KEYS & event.keys():
        event[name] = "***"
    return event


def _write_line(line: str) -> None:
    # Opened per event, not cached: LEPIKA_HOME can change between calls (tests do
    # this on every test), and a long-lived handle would keep writing to the old file.
    with (logs_dir() / LOG_FILE).open("a", encoding="utf-8") as f:
        f.write(line + "\n")


class _FileLogger:
    """The sink structlog renders into: every level is one appended line."""

    def msg(self, message: str) -> None:
        _write_line(message)

    log = debug = info = warning = error = critical = msg


structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        _redact,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=lambda *args: _FileLogger(),
    cache_logger_on_first_use=False,
)


def get_logger() -> Any:
    """A structlog logger bound to LePika's log file. Use `log.info("event", k=v)`."""
    return structlog.get_logger()
