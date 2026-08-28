"""Typer entry point for ezai."""

from __future__ import annotations

import importlib.metadata

import typer

app = typer.Typer(
    help="One command → local AI chat in your browser.",
    add_completion=False,
)


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
    app()
