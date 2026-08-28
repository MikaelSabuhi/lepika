"""Typer entry point for ezai."""

from __future__ import annotations

import importlib.metadata

import typer
from rich.console import Console
from rich.markup import escape

from ezai.errors import FriendlyError

app = typer.Typer(
    help="One command → local AI chat in your browser.",
    add_completion=False,
)

err_console = Console(stderr=True)


def _version_string() -> str:
    return f"ezai {importlib.metadata.version('ezai')}"


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        typer.echo(_version_string())
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo("Setup wizard coming soon. Run `ezai --help` for available commands.")


def run() -> None:
    """Console-script entry point."""
    try:
        app()
    except FriendlyError as exc:
        err_console.print(f"[red]✗ {escape(exc.problem)}[/red]")
        err_console.print(f"[yellow]→ {escape(exc.fix)}[/yellow]")
        raise SystemExit(1) from exc
