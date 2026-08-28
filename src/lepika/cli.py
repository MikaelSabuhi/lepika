"""Typer entry point for LePika."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import webbrowser
from types import ModuleType

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from lepika import config, detect, engine, express, log, models, paths, server
from lepika.errors import FriendlyError

app = typer.Typer(
    help="One command → local AI chat in your browser.",
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)

MODES = ("express", "server")


def _backend(cfg: config.Config) -> ModuleType:
    """The module that runs the stack in this mode — the only place the two differ.

    `express` and `server` deliberately expose the same start_stack/stop/update/logs
    surface, so every lifecycle command is written once against whichever is in play.
    """
    return server if cfg.mode == "server" else express


def _version_string() -> str:
    return f"lepika {importlib.metadata.version('lepika')}"


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what the wizard would do without doing it."
    ),
    mode: str | None = typer.Option(
        None, "--mode", help="express (default, no Docker) or server (docker compose)."
    ),
) -> None:
    if version:
        typer.echo(_version_string())
        raise typer.Exit()
    if mode is not None:
        if mode not in MODES:
            raise typer.BadParameter("--mode must be 'express' or 'server'.")
        if ctx.invoked_subcommand is not None:
            # The mode is chosen once, by the wizard, and then lives in config.toml;
            # accepting it here silently would look like it had switched something.
            raise typer.BadParameter(
                "--mode applies to the wizard only; run `lepika --mode server` (no subcommand)."
            )
    if ctx.invoked_subcommand is None:
        # Imported here, not at module scope: `wizard` imports `cli._open_browser`.
        from lepika import wizard

        wizard.run_wizard(dry_run=dry_run, mode=mode)


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
    cfg = config.load()
    console.print(detect.plan_sentence(info, cfg.mode))
    # A CPU-bound container on a GPU machine is a mystery worth one line up front.
    # `has_docker` first: gpu_note runs `docker info`, and without the binary that
    # would replace Server mode's Express hint with a generic "command not found".
    if cfg.mode == "server" and info.has_docker and (note := server.gpu_note(info)) is not None:
        console.print(f"[yellow]{escape(note)}[/yellow]")
    _ready(cfg, _backend(cfg).start_stack(info, cfg))


@app.command()
def down() -> None:
    """Stop the stack (Express: OpenWebUI only; Server: every container, models kept)."""
    cfg = config.load()
    if _backend(cfg).stop(detect.detect(), cfg):
        console.print("[green]✓ Stopped.[/green]")
    else:
        console.print("Nothing was running.")


@app.command()
def status() -> None:
    """Show what's running."""
    cfg = config.load()
    table = Table(title="lepika status")
    table.add_column("Service")
    table.add_column("State")
    # First row: every other row means something different in each mode.
    table.add_row("Mode", cfg.mode)
    vllm = server.vllm_active(cfg)
    engine_ok = (
        engine.vllm_up(server.VLLM_URL)
        if vllm
        else detect.api_up(cfg.engine_url, key=cfg.engine_key)
    )
    webui_ok = express.webui_up(cfg.webui_port)
    table.add_row(
        "Engine (vLLM)" if vllm else "Engine",
        "[green]up[/green]" if engine_ok else "[red]down[/red]",
    )
    # Which machine answered matters once the engine can live somewhere else.
    where = cfg.engine_url + ("" if cfg.engine_managed else " (remote)")
    if vllm:
        where = server.VLLM_URL
    table.add_row("Engine URL", escape(where))
    table.add_row("OpenWebUI", "[green]up[/green]" if webui_ok else "[red]down[/red]")
    table.add_row("Model", cfg.model or "[dim]not set[/dim]")
    console.print(table)


@app.command()
def logs(lines: int = typer.Option(50, min=1, help="Lines per log file.")) -> None:
    """Print the tail of LePika's logs (Server mode: the container logs too)."""
    sections = _backend(config.load()).logs(lines)
    if not sections:
        # Silence is indistinguishable from a broken command.
        console.print("(no logs yet)")
        return
    for title, text in sections:
        console.rule(title)
        console.print(text, markup=False)


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
    """Upgrade the engine and OpenWebUI to their latest versions."""
    console.print("Upgrading…")
    cfg = config.load()
    _backend(cfg).update(detect.detect(), cfg)
    console.print("[green]✓ Everything is up to date and running.[/green]")


def _apply_engine_change(cfg: config.Config) -> bool:
    """Make a running OpenWebUI use the engine just configured; did anything restart?

    OpenWebUI reads the engine URL and key from its environment once, at startup, so
    a UI that is already up keeps talking to the old engine no matter what the config
    says — and `lepika up` would not fix it in Express mode, because a healthy UI is
    left alone. This is what makes `lepika connect` take effect on a live stack.
    """
    if cfg.mode == "server":
        # `compose up -d` is the Server-mode reconciler: it rewrites .env and
        # recreates whatever the new engine URL changed, so it needs no probe.
        server.start_stack(detect.detect(), cfg)
        return True
    if not express.webui_up(cfg.webui_port):
        return False
    express.restart_openwebui(cfg, detect.detect().os)
    return True


@app.command()
def connect(
    url: str | None = typer.Argument(None, help="Engine URL, e.g. http://gpu-box:11435"),
    key: str = typer.Option("", "--key", help="API key, if the engine needs one."),
    local: bool = typer.Option(False, "--local", help="Go back to the engine on this machine."),
) -> None:
    """Use an engine running on another machine (or --local to stop doing so)."""
    cfg = config.load()
    if local:
        cfg.engine_managed, cfg.engine_url, cfg.engine_key = True, config.DEFAULT_ENGINE_URL, ""
        config.save(cfg)
        log.get_logger().info("engine.connect", url=cfg.engine_url, local=True)
        if _apply_engine_change(cfg):
            console.print("[green]✓ Using the local engine again[/green] — OpenWebUI restarted.")
        else:
            console.print("[green]✓ Using the local engine again.[/green] Run `lepika up`.")
        return
    if url is None:
        raise typer.BadParameter("Give an engine URL, or --local.")
    url = url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        # `gpu-box:11435` parses as a scheme, so every probe of it fails obscurely.
        raise FriendlyError(
            "Engine URLs start with http:// or https://",
            "Give the whole address, e.g. `lepika connect http://gpu-box:11435`.",
        )
    if not detect.api_up(url, key=key):
        raise FriendlyError(
            f"No engine answered at {url}.",
            "Check the address and that `lepika expose` is on over there "
            "(add --key if it printed one).",
        )
    cfg.engine_managed, cfg.engine_url, cfg.engine_key = False, url, key
    config.save(cfg)
    log.get_logger().info("engine.connect", url=url, key=key)
    if _apply_engine_change(cfg):
        console.print(f"[green]✓ Connected to[/green] {escape(url)} — OpenWebUI restarted.")
    else:
        console.print(f"[green]✓ Connected to[/green] {escape(url)}. Run `lepika up`.")


@app.command()
def expose(
    off: bool = typer.Option(False, "--off", help="Back to localhost only."),
    show: bool = typer.Option(False, "--show", help="Print the key and connect line again."),
    rotate: bool = typer.Option(False, "--rotate", help="Generate a new key."),
) -> None:
    """Share the engine (and the chat UI) with your network behind a generated API key."""
    cfg = config.load()
    if cfg.mode != "server":
        raise FriendlyError(
            "`lepika expose` needs Server mode (it runs a small Caddy proxy in the stack).",
            "Run `lepika --mode server` on this machine first.",
        )
    if off:
        cfg.exposed = False
        config.save(cfg)
        server.start_stack(detect.detect(), cfg)
        log.get_logger().info("expose.off")
        console.print("[green]✓ Back to localhost only.[/green]")
        return
    # Written before the stack starts: `start_stack` refuses to run the `expose`
    # profile with an empty key, because Caddy would then match a bare `Bearer `.
    key = server.api_key(paths.stack_dir() / server.ENV_FILE, rotate=rotate)
    # A rotation always restarts, --show or not: Caddy reads the key once, at
    # startup, so skipping the restart would leave the old key the working one.
    if not show or rotate:
        cfg.exposed = True
        config.save(cfg)
        server.start_stack(detect.detect(), cfg)
        log.get_logger().info("expose.on", api_port=cfg.api_port)
    ip = server.lan_ip()
    if cfg.exposed:
        console.print(f"[green]✓ Exposed.[/green] Chat UI: http://{ip}:{cfg.webui_port}")
    else:
        console.print("Not exposed yet — run `lepika expose` to turn it on.")
        console.print(f"Chat UI once it is on: http://{ip}:{cfg.webui_port}")
    console.print("The chat UI asks for a sign-in; the engine API wants the key below.")
    # soft_wrap on the copied lines: an 80-column wrap breaks a URL or a key.
    if server.vllm_active(cfg):
        # `lepika connect` probes Ollama's /api/version, which Caddy in front of vLLM
        # never answers — so point at the OpenAI-compatible URL OpenWebUI can add.
        console.print(f"Engine API is vLLM (OpenAI-compatible): http://{ip}:{cfg.api_port}/v1")
        console.print(
            "On another machine, add that URL with the key as an OpenAI connection in "
            "OpenWebUI (Admin → Settings → Connections); `lepika connect` is for "
            "Ollama engines."
        )
        console.print(f"  Key: {key}", markup=False, soft_wrap=True)
        if rotate:
            # Engine-neutral: reconnecting here is editing the OpenAI connection in
            # OpenWebUI, and `lepika connect` refuses a vLLM engine outright.
            console.print("Machines using the old key need the new one above.")
    else:
        console.print(f"Engine API: http://{ip}:{cfg.api_port}")
        console.print("On another machine:")
        console.print(
            f"  lepika connect http://{ip}:{cfg.api_port} --key {key}", markup=False, soft_wrap=True
        )
        if rotate:
            # Caddy only honours the new key, so every machine holding the old one is
            # locked out until it is told — silence there looks like a broken engine.
            console.print(
                f"Machines that connected before need `lepika connect http://{ip}:{cfg.api_port} "
                f"--key {key}` again.",
                markup=False,
                soft_wrap=True,
            )
    console.print("Keep the key private; `lepika expose --rotate` replaces it.")


model_app = typer.Typer(help="Add, list, or remove local models.")
app.add_typer(model_app, name="model")


def _refuse_ollama_only(cfg: config.Config, detail: str) -> None:
    """`model list`/`model rm` talk to Ollama, which a vLLM stack deliberately stops.

    Without this the engine looks unreachable and the hint is `lepika up` — which
    starts the stack it is already in, leaving the user in a loop.
    """
    if server.vllm_active(cfg):
        raise FriendlyError(
            f"'{cfg.model}' is served by vLLM — {detail}",
            "Switch models with `lepika model add <ref>`; vLLM keeps its weights in "
            "the hf-cache volume.",
        )


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
    cfg = config.load()
    if ref is None:
        model_ref = wizard.choose_model(info, cfg)
    else:
        # Same rejection as the wizard's, by reusing it rather than restating it.
        model_ref = wizard._validate(models.parse_model_ref(ref), cfg, info)
    if models.uses_vllm(model_ref.raw):
        # Nothing to pull: vLLM downloads the weights itself when it starts, and it
        # only starts once the config names the model compose interpolates.
        server.hf_token_prompt(paths.stack_dir() / server.ENV_FILE)
        cfg.model = model_ref.raw
        config.save(cfg)
        console.print(f"[green]✓ Set:[/green] {escape(model_ref.raw)} — starting vLLM…")
        _ready(cfg, server.start_stack(info, cfg))
        return
    if cfg.engine_managed and cfg.mode == "server":
        # In Server mode the engine is a container: it has to be up before there is
        # anything to pull into — and it has to be the engine the *new* model needs.
        # Switching away from a full-weight repo with the old ref still in the config
        # would bring up vLLM, leave ollama stopped, and fail the pull.
        server.start_stack(info, dataclasses.replace(cfg, model=model_ref.raw))
    elif cfg.engine_managed:
        express.ensure_ollama(info, url=cfg.engine_url)
    else:
        # A remote engine is someone else's to run: check it, never install for it.
        express.check_remote_engine(cfg, api_up=detect.api_up)
    engine.pull_model(cfg.engine_url, model_ref, key=cfg.engine_key)
    cfg.model = model_ref.raw
    config.save(cfg)
    console.print(f"[green]✓ Added:[/green] {escape(model_ref.raw)}")


@model_app.command("list")
def model_list() -> None:
    """List downloaded models."""
    cfg = config.load()
    _refuse_ollama_only(cfg, "there is no Ollama model list to show.")
    installed = engine.list_models(cfg.engine_url, key=cfg.engine_key)
    if not installed:
        console.print("No models yet — run `lepika model add`.", markup=False)
        return
    table = Table(title="models")
    table.add_column("Name")
    table.add_column("Size", justify="right")
    for name, size in installed:
        marker = " [dim](default)[/dim]" if name == cfg.model else ""
        table.add_row(escape(name) + marker, engine.human_size(size))
    console.print(table)


@model_app.command("rm")
def model_rm(
    name: str = typer.Argument(..., help="Model name as shown by `lepika model list`."),
) -> None:
    """Remove a downloaded model."""
    cfg = config.load()
    _refuse_ollama_only(cfg, "there is nothing to remove from Ollama.")
    engine.delete_model(cfg.engine_url, name, key=cfg.engine_key)
    console.print(f"[green]✓ Removed:[/green] {escape(name)}")


def run() -> None:
    """Console-script entry point."""
    try:
        app()
    except FriendlyError as exc:
        err_console.print(f"[red]✗ {escape(exc.problem)}[/red]")
        err_console.print(f"[yellow]→ {escape(exc.fix)}[/yellow]")
        raise SystemExit(1) from exc
