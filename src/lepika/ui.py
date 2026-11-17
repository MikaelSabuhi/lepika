"""Everything LePika shows or asks goes through here.

Output keeps going through two real rich consoles (stdout and stderr) that follow
`sys.stdout`/`sys.stderr` at write time, so what a test captures is what rich
renders. Questions have one "no terminal" policy: piped, redirected, under cron or
in CI, rich's prompts read EOF from stdin. A question with a default answers itself
with that default and says so in one dim line; a question without one cannot be
guessed, so it becomes a `NoTerminal` (a `FriendlyError`) rather than a traceback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, Prompt

from lepika.errors import FriendlyError

Reader = Callable[..., Any]

_console = Console()
_err_console = Console(stderr=True)


class NoTerminal(FriendlyError):
    """A question with no default met EOF: there is nobody to answer it."""


def _no_terminal(prompt: str) -> NoTerminal:
    return NoTerminal(
        f"“{prompt}” needs an answer, and there is no terminal to ask.",
        "Run the command again from an interactive terminal.",
    )


class UI:
    """Everything LePika shows or asks.

    `reader` and `confirm_reader` answer `ask` and `confirm`; None means rich's own
    `Prompt.ask`/`Confirm.ask`, looked up when the question is asked.
    """

    def __init__(
        self, *, reader: Reader | None = None, confirm_reader: Reader | None = None
    ) -> None:
        self._reader = reader
        self._confirm_reader = confirm_reader

    def say(self, *objects: Any, markup: bool | None = None, soft_wrap: bool | None = None) -> None:
        """Print to stdout, exactly as `Console.print` would."""
        _console.print(*objects, markup=markup, soft_wrap=soft_wrap)

    def say_error(self, *objects: Any) -> None:
        """Print to stderr, exactly as `Console.print` would."""
        _err_console.print(*objects)

    def ask(
        self,
        prompt: str,
        *,
        default: str | None = None,
        password: bool = False,
        require_terminal: bool = False,
    ) -> str:
        """Ask once. `require_terminal`: Enter still takes the default, but no terminal
        raises `NoTerminal` — for a caller whose no-terminal answer needs its own words."""
        # Resolved at call time, not import time, so a test patching `Prompt.ask` reaches it.
        read: Reader = self._reader if self._reader is not None else Prompt.ask
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
                self.say("[dim]No terminal to ask — skipping.[/dim]")
            else:
                self.say(f"[dim]No terminal to ask — using {escape(default)}.[/dim]")
            return default

    def confirm(
        self,
        prompt: str,
        *,
        default: bool | None = None,
        require_terminal: bool = False,
    ) -> bool:
        """Ask yes/no once; `require_terminal` as for `ask`."""
        read: Reader = self._confirm_reader if self._confirm_reader is not None else Confirm.ask
        kwargs: dict[str, Any] = {}
        if default is not None:
            kwargs["default"] = default
        try:
            return bool(read(prompt, **kwargs))
        except EOFError as exc:
            if default is None or require_terminal:
                raise _no_terminal(prompt) from exc
            self.say(f"[dim]No terminal to ask — using {'yes' if default else 'no'}.[/dim]")
            return default


# The one the CLI hands down; looked up when used, never bound, so a test can replace it.
REAL = UI()
