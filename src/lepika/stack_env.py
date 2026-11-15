"""The stack's .env: the file compose reads, which holds the API key and an HF token."""

from __future__ import annotations

import secrets
from pathlib import Path

from lepika import fsio, log

ENV_FILE = ".env"


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
