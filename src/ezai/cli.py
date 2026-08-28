"""Typer entry point for ezai."""

from __future__ import annotations

import importlib.metadata
import webbrowser

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ezai import config, detect, express, models, paths, proc
from ezai.errors import FriendlyError

app = typer.Typer(
    help="One command → local AI chat in your browser.",
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)


def _version_string() -> str:
    return f"ezai {importlib.metadata.version('ezai')}"


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what the wizard would do without doing it."
    ),
) -> None:
    if version:
        typer.echo(_version_string())
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        # Imported here, not at module scope: `wizard` imports `cli._open_browser`.
        from ezai import wizard

        wizard.run_wizard(dry_run=dry_run)


def _open_browser(url: str) -> None:
    webbrowser.open(url)


@app.command()
def up() -> None:
    """Start the local AI stack and open the browser."""
    info = detect.detect()
    console.print(detect.plan_sentence(info))
    cfg = config.load()
    express.ensure_ollama(info)
    express.ensure_openwebui(cfg)
    url = express.webui_url(cfg.webui_port)
    console.print(f"[green]✓ Ready:[/green] {url}")
    _open_browser(url)


@app.command()
def down() -> None:
    """Stop OpenWebUI (Ollama keeps running as a shared service)."""
    info = detect.detect()
    if express.stop_openwebui(info.os):
        console.print("[green]✓ OpenWebUI stopped.[/green]")
    else:
        console.print("OpenWebUI was not running.")


@app.command()
def status() -> None:
    """Show what's running."""
    cfg = config.load()
    table = Table(title="ezai status")
    table.add_column("Service")
    table.add_column("State")
    ollama_ok = detect.api_up(detect.OLLAMA_URL)
    webui_ok = express.webui_up(cfg.webui_port)
    table.add_row("Ollama API", "[green]up[/green]" if ollama_ok else "[red]down[/red]")
    table.add_row("OpenWebUI", "[green]up[/green]" if webui_ok else "[red]down[/red]")
    table.add_row("Model", cfg.model or "[dim]not set[/dim]")
    console.print(table)


@app.command()
def logs(lines: int = typer.Option(50, help="Lines per log file.")) -> None:
    """Print the tail of ezai's log files."""
    for log_file in sorted(paths.logs_dir().glob("*.log")):
        console.rule(str(log_file.name))
        content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in content[-lines:]:
            console.print(line, markup=False)


model_app = typer.Typer(help="Add, list, or remove local models.")
app.add_typer(model_app, name="model")


@model_app.command("add")
def model_add(
    ref: str | None = typer.Argument(
        None, help="qwen3:8b · hf.co/<org>/<repo>-GGUF · leave empty to browse"
    ),
) -> None:
    """Download a model and make it the default."""
    # Imported here, not at module scope: `wizard` imports `cli._open_browser`.
    from ezai import wizard

    info = detect.detect()
    if ref is None:
        model_ref = wizard.choose_model(info)
    else:
        # Same rejection as the wizard's, by reusing it rather than restating it.
        model_ref = wizard._validate(models.parse_model_ref(ref))
    express.ensure_ollama(info)
    express.pull_model(model_ref)
    cfg = config.load()
    cfg.model = model_ref.raw
    config.save(cfg)
    console.print(f"[green]✓ Added:[/green] {escape(model_ref.raw)}")


@model_app.command("list")
def model_list() -> None:
    """List downloaded models."""
    result = proc.run_logged(["ollama", "list"], check=False)
    console.print(result.stdout or "No models yet — run `ezai model add`.", markup=False)


@model_app.command("rm")
def model_rm(
    name: str = typer.Argument(..., help="Model name as shown by `ezai model list`."),
) -> None:
    """Remove a downloaded model."""
    proc.run_logged(["ollama", "rm", name])
    console.print(f"[green]✓ Removed:[/green] {escape(name)}")


def run() -> None:
    """Console-script entry point."""
    try:
        app()
    except FriendlyError as exc:
        err_console.print(f"[red]✗ {escape(exc.problem)}[/red]")
        err_console.print(f"[yellow]→ {escape(exc.fix)}[/yellow]")
        raise SystemExit(1) from exc
