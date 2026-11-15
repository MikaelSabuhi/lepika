"""Every interactive question goes through here, so "no terminal" has one policy.

Piped, redirected, under cron or in CI, rich's prompts read EOF from stdin. A
question with a default answers itself with that default and says so in one dim
line; a question without one cannot be guessed, so it becomes a `NoTerminal`
(a `FriendlyError`) rather than a traceback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, Prompt

from lepika.errors import FriendlyError

Reader = Callable[..., Any]

console = Console()


class NoTerminal(FriendlyError):
    """A question with no default met EOF: there is nobody to answer it."""


def _no_terminal(prompt: str) -> NoTerminal:
    return NoTerminal(
        f"“{prompt}” needs an answer, and there is no terminal to ask.",
        "Run the command again from an interactive terminal.",
    )


def ask(
    prompt: str,
    *,
    default: str | None = None,
    password: bool = False,
    reader: Reader | None = None,
    require_terminal: bool = False,
) -> str:
    """Ask once. `require_terminal`: Enter still takes the default, but no terminal
    raises `NoTerminal` — for a caller whose no-terminal answer needs its own words."""
    # Resolved at call time, not import time, so a test patching `Prompt.ask` reaches it.
    read: Reader = reader if reader is not None else Prompt.ask
    # The same keywords rich takes, each only when it says something: rich treats a
    # passed default as "Enter means this", which an absent one must not become, and
    # the call a reader sees stays exactly what it was before this module existed.
    kwargs: dict[str, Any] = {}
    if password:
        kwargs["password"] = password
    if default is not None:
        kwargs["default"] = default
    try:
        return str(read(prompt, **kwargs))
    except EOFError as exc:
        if default is None or require_terminal:
            raise _no_terminal(prompt) from exc
        if password or not default:
            # A secret's default is never echoed, and an empty one is a skip.
            console.print("[dim]No terminal to ask — skipping.[/dim]")
        else:
            console.print(f"[dim]No terminal to ask — using {escape(default)}.[/dim]")
        return default


def confirm(
    prompt: str,
    *,
    default: bool | None = None,
    reader: Reader | None = None,
    require_terminal: bool = False,
) -> bool:
    """Ask yes/no once; `require_terminal` as for `ask`."""
    read: Reader = reader if reader is not None else Confirm.ask
    kwargs: dict[str, Any] = {}
    if default is not None:
        kwargs["default"] = default
    try:
        return bool(read(prompt, **kwargs))
    except EOFError as exc:
        if default is None or require_terminal:
            raise _no_terminal(prompt) from exc
        console.print(f"[dim]No terminal to ask — using {'yes' if default else 'no'}.[/dim]")
        return default
