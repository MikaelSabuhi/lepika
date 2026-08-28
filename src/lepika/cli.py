"""Typer entry point for LePika."""

from __future__ import annotations

import importlib.metadata
import shutil
import webbrowser

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from lepika import config, detect, express, models, paths, proc
from lepika.errors import FriendlyError

app = typer.Typer(
    help="One command → local AI chat in your browser.",
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)


def _version_string() -> str:
    return f"lepika {importlib.metadata.version('lepika')}"


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
        from lepika import wizard

        wizard.run_wizard(dry_run=dry_run)


def _open_browser(url: str) -> None:
    webbrowser.open(url)


def _ready(cfg: config.Config, url: str) -> None:
    """Announce a running stack and open it — shared by `lepika up` and the wizard."""
    console.print(f"[green]✓ Ready:[/green] {url}")
    if not cfg.model:
        # A chat UI with no model behind it looks broken; say what's missing.
        console.print("No model yet — run `lepika` or `lepika model add`.")
    _open_browser(url)


@app.command()
def up() -> None:
    """Start the local AI stack and open the browser."""
    info = detect.detect()
    console.print(detect.plan_sentence(info))
    cfg = config.load()
    _ready(cfg, express.start_stack(info, cfg))


@app.command()
def down() -> None:
    """Stop OpenWebUI (Ollama keeps running as a shared service)."""
    info = detect.detect()
    # The port is what proves the recorded pid is still our OpenWebUI.
    if express.stop_openwebui(info.os, port=config.load().webui_port):
        console.print("[green]✓ OpenWebUI stopped.[/green]")
    else:
        console.print("OpenWebUI was not running.")


@app.command()
def status() -> None:
    """Show what's running."""
    cfg = config.load()
    table = Table(title="lepika status")
    table.add_column("Service")
    table.add_column("State")
    ollama_ok = detect.api_up(cfg.engine_url)
    webui_ok = express.webui_up(cfg.webui_port)
    table.add_row("Ollama API", "[green]up[/green]" if ollama_ok else "[red]down[/red]")
    table.add_row("OpenWebUI", "[green]up[/green]" if webui_ok else "[red]down[/red]")
    table.add_row("Model", cfg.model or "[dim]not set[/dim]")
    console.print(table)


@app.command()
def logs(lines: int = typer.Option(50, min=1, help="Lines per log file.")) -> None:
    """Print the tail of LePika's log files."""
    log_files = sorted(paths.logs_dir().glob("*.log"))
    if not log_files:
        # Silence is indistinguishable from a broken command.
        console.print("(no logs yet)")
        return
    for log_file in log_files:
        console.rule(str(log_file.name))
        content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in content[-lines:]:
            console.print(line, markup=False)


@app.command()
def doctor() -> None:
    """Diagnose the local setup."""
    # Imported here, not at module scope: this command function shadows the name.
    from lepika import doctor as doctor_mod

    info = detect.detect()
    results = doctor_mod.run_checks(info)
    core_failed = False
    for r in results:
        mark = "[green]✓[/green]" if r.ok else "[red]✗[/red]"
        console.print(f"{mark} {r.name}")
        if not r.ok:
            console.print(f"  [yellow]→ {escape(r.hint)}[/yellow]")
            if r.name != doctor_mod.RAM_CHECK:
                core_failed = True
    if core_failed:
        raise typer.Exit(code=1)


@app.command()
def update() -> None:
    """Upgrade Ollama and OpenWebUI to their latest versions."""
    info = detect.detect()
    console.print("Upgrading Ollama…")
    if info.os == "macos":
        if shutil.which("brew") is not None:
            # check=False: brew exits nonzero when ollama is already up to date.
            proc.run_logged(["brew", "upgrade", "ollama"], check=False)
        else:
            console.print("Ollama.app updates itself — skipping engine upgrade.")
    elif info.os == "linux":
        # Re-running the official script upgrades in place. Reused rather than
        # restated: it must stream, because it may prompt for sudo.
        express.install_ollama(info)
    else:
        # check=False: winget exits nonzero when no upgrade is available.
        proc.run_logged(["winget", "upgrade", "--id", "Ollama.Ollama", "-e"], check=False)
    console.print("Upgrading OpenWebUI…")
    # check=False: uv exits nonzero when open-webui is already the latest version.
    proc.run_logged(["uv", "tool", "upgrade", "open-webui"], check=False)
    # A restart, not a stop-then-probe: the upgraded build only takes effect once
    # the old server is really gone.
    express.restart_openwebui(config.load(), info.os)
    console.print("[green]✓ Everything is up to date and running.[/green]")


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
    from lepika import wizard

    info = detect.detect()
    if ref is None:
        model_ref = wizard.choose_model(info)
    else:
        # Same rejection as the wizard's, by reusing it rather than restating it.
        model_ref = wizard._validate(models.parse_model_ref(ref))
    cfg = config.load()
    express.ensure_ollama(info, url=cfg.engine_url)
    express.pull_model(model_ref)
    cfg.model = model_ref.raw
    config.save(cfg)
    console.print(f"[green]✓ Added:[/green] {escape(model_ref.raw)}")


@model_app.command("list")
def model_list() -> None:
    """List downloaded models."""
    result = proc.run_logged(["ollama", "list"], check=False, log=False)
    if result.returncode != 0:
        # An unreachable engine looks identical to an empty list without this.
        console.print("Could not reach Ollama — run `lepika doctor`.", markup=False)
        raise typer.Exit(code=1)
    console.print(result.stdout or "No models yet — run `lepika model add`.", markup=False)


@model_app.command("rm")
def model_rm(
    name: str = typer.Argument(..., help="Model name as shown by `lepika model list`."),
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
