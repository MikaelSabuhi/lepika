"""Typer entry point for LePika."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.metadata
import io
import os
import sys
import traceback
import webbrowser
from collections.abc import Iterable
from pathlib import Path
from typing import NoReturn, TextIO

import typer
from rich.markup import escape
from rich.rule import Rule
from rich.table import Table

from lepika import (
    acquire,
    backend,
    config,
    detect,
    engine,
    health,
    log,
    models,
    ollama,
    openwebui,
    paths,
    server,
    serving,
    stack_env,
    system,
    ui,
    wizard,
)
from lepika.errors import ISSUES_URL, CommandNotFound, FriendlyError
from lepika.updates import UpdateStep

app = typer.Typer(
    help="One command → local AI chat in your browser.",
    add_completion=False,
)


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
        if mode not in config.MODES:
            raise typer.BadParameter("--mode must be 'express' or 'server'.")
        if ctx.invoked_subcommand is not None:
            # The mode is chosen once, by the wizard, and then lives in config.toml;
            # accepting it here silently would look like it had switched something.
            raise typer.BadParameter(
                "--mode applies to the wizard only; run `lepika --mode server` (no subcommand)."
            )
    if ctx.invoked_subcommand is None:
        result = wizard.run_wizard(dry_run=dry_run, mode=mode, sys=system.REAL, ui=ui.REAL)
        if result is not None:
            _ready(*result)


def _open_browser(url: str) -> None:
    webbrowser.open(url)


def _ready(cfg: config.Config, url: str) -> None:
    """Announce a running stack and open it — shared by `lepika up` and the wizard."""
    ui.REAL.say(f"[green]✓ Ready:[/green] {url}")
    if not cfg.model:
        # A chat UI with no model behind it looks broken; say what's missing.
        ui.REAL.say("No model yet — run `lepika` or `lepika model add`.")
    _open_browser(url)


@app.command()
def up() -> None:
    """Start the local AI stack and open the browser."""
    info = detect.detect(sys=system.REAL)
    cfg = config.load()
    ui.REAL.say(detect.plan_sentence(info, cfg.mode, engine=serving.engine_label(cfg)))
    # A CPU-bound container on a GPU machine is a mystery worth one line up front.
    # Docker's presence first: gpu_note runs `docker info`, and without the binary that
    # would replace Server mode's Express hint with a generic "command not found".
    if (
        cfg.mode == "server"
        and system.REAL.which("docker") is not None
        and (note := server.gpu_note(info, sys=system.REAL)) is not None
    ):
        ui.REAL.say(f"[yellow]{escape(note)}[/yellow]")
    # Asked before the start, and that order is the point: an engine answering
    # now is one the start will adopt, so config.toml's context length cannot reach it.
    if (note := ollama.context_note(info, cfg, sys=system.REAL)) is not None:
        ui.REAL.say(f"[yellow]{escape(note)}[/yellow]")
    _ready(cfg, backend.backend_for(cfg, sys=system.REAL, ui=ui.REAL).start(info, cfg))


@app.command()
def down() -> None:
    """Stop the stack (Express: OpenWebUI only; Server: every container, models kept)."""
    cfg = config.load()
    if backend.backend_for(cfg, sys=system.REAL, ui=ui.REAL).stop(
        detect.detect(sys=system.REAL), cfg
    ):
        ui.REAL.say("[green]✓ Stopped.[/green]")
    else:
        ui.REAL.say("Nothing was running.")


@app.command()
def status() -> None:
    """Show what's running."""
    cfg = config.load()
    table = Table(title="lepika status")
    table.add_column("Service")
    table.add_column("State")
    # First row: every other row means something different in each mode.
    table.add_row("Mode", cfg.mode)
    vllm = serving.vllm_active(cfg)
    engine_ok = (
        health.vllm_up(server.VLLM_URL, sys=system.REAL)
        if vllm
        else health.api_up(cfg.engine_url, key=cfg.engine_key, sys=system.REAL)
    )
    webui_ok = health.webui_up(cfg.webui_port, sys=system.REAL)
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
    if cfg.mode == "server":
        # Only Server mode can expose; a row that can only say "no" is noise in Express.
        table.add_row(
            "Exposed",
            f"[yellow]yes[/yellow] — API on port {cfg.api_port}" if cfg.exposed else "no",
        )
    table.add_row("Model", cfg.model or "[dim]not set[/dim]")
    ui.REAL.say(table)


@app.command()
def logs(lines: int = typer.Option(50, min=1, help="Lines per log file.")) -> None:
    """Print the tail of LePika's logs (Server mode: the container logs too)."""
    sections = backend.backend_for(config.load(), sys=system.REAL, ui=ui.REAL).logs(lines)
    if not sections:
        # Silence is indistinguishable from a broken command.
        ui.REAL.say("(no logs yet)")
        return
    for title, text in sections:
        ui.REAL.say(Rule(title))
        ui.REAL.say(text, markup=False)


@app.command()
def doctor() -> None:
    """Diagnose the local setup."""
    # Imported here, not at module scope: this command function shadows the name.
    from lepika import doctor as doctor_mod

    info = detect.detect(sys=system.REAL)
    results = doctor_mod.run_checks(info, sys=system.REAL)
    core_failed = False
    for r in results:
        mark = "[green]✓[/green]" if r.ok else "[red]✗[/red]"
        ui.REAL.say(f"{mark} {r.name}")
        if not r.ok:
            ui.REAL.say(f"  [yellow]→ {escape(r.hint)}[/yellow]")
            if r.name != doctor_mod.RAM_CHECK:
                core_failed = True
    if core_failed:
        raise typer.Exit(code=1)


@app.command()
def update() -> None:
    """Upgrade the engine and OpenWebUI to their latest versions."""
    ui.REAL.say("Upgrading…")
    cfg = config.load()
    steps: list[UpdateStep] = backend.backend_for(cfg, sys=system.REAL, ui=ui.REAL).update(
        detect.detect(sys=system.REAL), cfg
    )
    for step in steps:
        detail = escape(step.detail)
        if step.outcome in ("stale", "note"):
            ui.REAL.say(f"[yellow]! {step.component}: {detail}[/yellow]")
            continue
        suffix = f" ({detail})" if detail else ""
        verb = "upgraded" if step.outcome == "upgraded" else "up to date"
        ui.REAL.say(f"[green]✓ {step.component}: {verb}{suffix}[/green]")


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
        backend.backend_for(cfg, sys=system.REAL, ui=ui.REAL).start(
            detect.detect(sys=system.REAL), cfg
        )
        return True
    if not health.webui_up(cfg.webui_port, sys=system.REAL):
        return False
    openwebui.restart_openwebui(cfg, detect.detect(sys=system.REAL).os, sys=system.REAL)
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
            ui.REAL.say("[green]✓ Using the local engine again[/green] — OpenWebUI restarted.")
        else:
            ui.REAL.say("[green]✓ Using the local engine again.[/green] Run `lepika up`.")
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
    if key and cfg.exposed:
        # The mirror of `expose`'s refusal: Caddy in front of us forwards the request
        # with the key it generates here, which the keyed remote rejects. Connecting
        # first and exposing later is the same 401-only proxy, built the other way round.
        raise FriendlyError(
            f"This machine is exposed, and the proxy can't relay a keyed engine at {url}.",
            "Run `lepika expose --off` first, then connect again — or connect without --key.",
        )
    if not health.api_up(url, key=key, sys=system.REAL):
        raise FriendlyError(
            f"No engine answered at {url}.",
            "Check the address and that `lepika expose` is on over there "
            "(add --key if it printed one).",
        )
    cfg.engine_managed, cfg.engine_url, cfg.engine_key = False, url, key
    config.save(cfg)
    log.get_logger().info("engine.connect", url=url, key=key)
    if _apply_engine_change(cfg):
        ui.REAL.say(f"[green]✓ Connected to[/green] {escape(url)} — OpenWebUI restarted.")
    else:
        ui.REAL.say(f"[green]✓ Connected to[/green] {escape(url)}. Run `lepika up`.")


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
        backend.backend_for(cfg, sys=system.REAL, ui=ui.REAL).start(
            detect.detect(sys=system.REAL), cfg
        )
        log.get_logger().info("expose.off")
        ui.REAL.say("[green]✓ Back to localhost only.[/green]")
        return
    if cfg.engine_key:
        # Caddy checks LePika's key and then forwards the request — that same
        # header included — to the upstream. A remote engine with its own key
        # rejects every one of them, so the exposed API would only ever say 401.
        raise FriendlyError(
            f"`lepika expose` can't relay {cfg.engine_url}: that engine needs its own key, "
            "and the proxy passes on the one it generates here.",
            "Point other machines at that engine directly (`lepika expose --show` there), "
            "or `lepika connect --local` first and expose an engine this machine runs.",
        )
    # Written before the stack starts: the start refuses to run the `expose`
    # profile with an empty key, because Caddy would then match a bare `Bearer `.
    key = stack_env.api_key(paths.stack_dir() / stack_env.ENV_FILE, rotate=rotate)
    # A rotation always restarts, --show or not: Caddy reads the key once, at
    # startup, so skipping the restart would leave the old key the working one.
    if not show or rotate:
        cfg.exposed = True
        config.save(cfg)
        backend.backend_for(cfg, sys=system.REAL, ui=ui.REAL).start(
            detect.detect(sys=system.REAL), cfg
        )
        log.get_logger().info("expose.on", api_port=cfg.api_port)
    ips = server.lan_ips(sys=system.REAL)
    # Nothing resolved at all (offline, or a locked-down host): say so in words
    # rather than print something that reads like an address.
    ip = ips[0] if ips else server.NO_ROUTE
    if cfg.exposed:
        ui.REAL.say(f"[green]✓ Exposed.[/green] Chat UI: http://{ip}:{cfg.webui_port}")
    else:
        ui.REAL.say("Not exposed yet — run `lepika expose` to turn it on.")
        ui.REAL.say(f"Chat UI once it is on: http://{ip}:{cfg.webui_port}")
    if len(ips) > 1:
        # The route pick is the address this box would dial out on, which is not
        # always the one the laptop in the next room can reach.
        others = ", ".join(f"http://{other}:{cfg.webui_port}" for other in ips[1:])
        ui.REAL.say(f"Also reachable on: {others}", markup=False, soft_wrap=True)
    ui.REAL.say("The chat UI asks for a sign-in; the engine API wants the key below.")
    # soft_wrap on the copied lines: an 80-column wrap breaks a URL or a key.
    if serving.vllm_active(cfg):
        # `lepika connect` probes Ollama's /api/version, which Caddy in front of vLLM
        # never answers — so point at the OpenAI-compatible URL OpenWebUI can add.
        ui.REAL.say(f"Engine API is vLLM (OpenAI-compatible): http://{ip}:{cfg.api_port}/v1")
        ui.REAL.say(
            "On another machine, add that URL with the key as an OpenAI connection in "
            "OpenWebUI (Admin → Settings → Connections); `lepika connect` is for "
            "Ollama engines."
        )
        ui.REAL.say(f"  Key: {key}", markup=False, soft_wrap=True)
        if rotate:
            # Engine-neutral: reconnecting here is editing the OpenAI connection in
            # OpenWebUI, and `lepika connect` refuses a vLLM engine outright.
            ui.REAL.say("Machines using the old key need the new one above.")
    else:
        ui.REAL.say(f"Engine API: http://{ip}:{cfg.api_port}")
        ui.REAL.say("On another machine:")
        ui.REAL.say(
            f"  lepika connect http://{ip}:{cfg.api_port} --key {key}", markup=False, soft_wrap=True
        )
        if rotate:
            # Caddy only honours the new key, so every machine holding the old one is
            # locked out until it is told — silence there looks like a broken engine.
            ui.REAL.say(
                f"Machines that connected before need `lepika connect http://{ip}:{cfg.api_port} "
                f"--key {key}` again.",
                markup=False,
                soft_wrap=True,
            )
    ui.REAL.say("Keep the key private; `lepika expose --rotate` replaces it.")


model_app = typer.Typer(help="Add, list, or remove local models.")
app.add_typer(model_app, name="model")


def _refuse_ollama_only(cfg: config.Config, detail: str) -> None:
    """`model list`/`model rm` talk to Ollama, which a vLLM stack deliberately stops.

    Without this the engine looks unreachable and the hint is `lepika up` — which
    starts the stack it is already in, leaving the user in a loop.
    """
    if serving.vllm_active(cfg):
        raise FriendlyError(
            f"'{cfg.model}' is served by vLLM — {detail}",
            "Switch models with `lepika model add <ref>`; vLLM keeps its weights in "
            "the hf-cache volume.",
        )


def _checked_quant(quant: str) -> str:
    """Refuse a `-q` Ollama's MLX runner would only reject minutes into a build."""
    if quant not in engine.IMPORT_QUANTS:
        raise typer.BadParameter(f"--quant must be one of {', '.join(engine.IMPORT_QUANTS)}.")
    return quant


# One description for both import paths: `model add <org>/<repo>` and `model import <dir>`.
QUANT_OPTION = typer.Option(
    engine.IMPORT_QUANT,
    "--quant",
    help="Quantization for a full-weight import: nvfp4 (default) or int4.",
)
# Out here rather than in the signature: ruff's B008 allows a call in a default only
# when the annotation is a known-immutable type, and `Path` is not one.
WEIGHTS_ARGUMENT = typer.Argument(..., help="Folder holding config.json and *.safetensors")


def _default_name(path: Path) -> str:
    """What Ollama should call a folder the user did not name.

    `path.name` first: it keeps the name as typed when the folder is a symlink into a
    cache snapshot. But it is empty for `.` and `/`, and literally `..` for a parent
    reference — `..` is spellable as an Ollama model name, so without this it would
    be built and saved as the default. Neither names a model: resolve to the real one.
    """
    return path.name if path.name not in ("", "..") else path.resolve().name


@model_app.command("add")
def model_add(
    ref: str | None = typer.Argument(
        None,
        help=(
            "qwen3:8b · hf.co/<org>/<repo>-GGUF · <org>/<repo> (full weights) · "
            "leave empty to browse"
        ),
    ),
    quant: str = QUANT_OPTION,
) -> None:
    """Download a model and make it the default."""
    # Checked before anything is detected or downloaded. Not refused on a pull, though:
    # an `hf.co/…` ref can still fall through to an import, so the flag is not wrong
    # there — it is simply ignored when the pull succeeds.
    quant = _checked_quant(quant)
    info = detect.detect(sys=system.REAL)
    cfg = config.load()
    if ref is None:
        model_ref = wizard.choose_model(info, cfg, sys=system.REAL, ui=ui.REAL)
    else:
        # Same rejection — and the same quant picker — as the wizard's, by reusing it.
        model_ref = wizard.resolve(
            models.parse_model_ref(ref), cfg, info, sys=system.REAL, ui=ui.REAL
        )
    if models.uses_vllm(model_ref.raw) and serving.vllm_allowed(cfg, info):
        # Nothing to pull: vLLM downloads the weights itself when it starts, and it
        # only starts once the config names the model compose interpolates.
        server.hf_token_prompt(paths.stack_dir() / stack_env.ENV_FILE, sys=system.REAL, ui=ui.REAL)
        cfg.model = model_ref.raw
        config.save(cfg)
        ui.REAL.say(f"[green]✓ Set:[/green] {escape(model_ref.raw)} — starting vLLM…")
        _ready(cfg, backend.backend_for(cfg, sys=system.REAL, ui=ui.REAL).start(info, cfg))
        return
    if cfg.engine_managed and cfg.mode == "server":
        # In Server mode the engine is a container: it has to be up before there is
        # anything to pull into — and it has to be the engine the *new* model needs.
        # Switching away from a full-weight repo with the old ref still in the config
        # would bring up vLLM, leave ollama stopped, and fail the pull.
        backend.backend_for(cfg, sys=system.REAL, ui=ui.REAL).start(
            info, dataclasses.replace(cfg, model=model_ref.raw)
        )
    elif cfg.engine_managed:
        ollama.ensure_ollama(
            info, url=cfg.engine_url, context_length=cfg.context_length, sys=system.REAL
        )
    else:
        # A remote engine is someone else's to run: check it, never install for it.
        health.check_remote_engine(cfg, sys=system.REAL)
    served = acquire.acquire(info, cfg, model_ref, quant=quant, sys=system.REAL, ui=ui.REAL)
    if served is None:
        ui.REAL.say("Nothing added.")
        return
    cfg.model = served
    config.save(cfg)
    ui.REAL.say(f"[green]✓ Added:[/green] {escape(served)}")


@model_app.command("import")
def model_import(
    path: Path = WEIGHTS_ARGUMENT,
    name: str = typer.Option("", "--name", help="What Ollama calls it (default: the folder name)."),
    quant: str = QUANT_OPTION,
) -> None:
    """Import safetensors weights already on disk into Ollama."""
    quant = _checked_quant(quant)
    info = detect.detect(sys=system.REAL)
    cfg = config.load()
    if not serving.import_allowed(cfg, info):
        # Before any engine call: a Server-mode config must not have an Ollama
        # installed or started for a model this machine could never build.
        raise FriendlyError(
            "Importing weights needs Express mode with an engine on this machine "
            "(Apple Silicon, or an NVIDIA GPU on 64-bit Linux/Windows).",
            "Run `lepika --mode express` on such a machine, or use a GGUF build: "
            "`lepika model add hf.co/<org>/<repo>-GGUF`.",
        )
    # Before the engine: a typo'd path is no reason to install Ollama on a machine that
    # does not have it. `check_local` is pure, so `import_local` runs it again.
    name = name or _default_name(path)
    acquire.check_local(path, name)
    # `import_allowed` already means Express with a managed engine, so there is no
    # container or remote case to branch on the way `model add` has to.
    ollama.ensure_ollama(
        info, url=cfg.engine_url, context_length=cfg.context_length, sys=system.REAL
    )
    # Resolved from here on: the "Importing … from …" line and the `source=` log field
    # name the folder unambiguously, whatever the shell was sitting in.
    served = acquire.import_local(
        info, cfg, path.resolve(), name, quant, sys=system.REAL, ui=ui.REAL
    )
    cfg.model = served
    config.save(cfg)
    ui.REAL.say(f"[green]✓ Imported:[/green] {escape(served)}")


@model_app.command("list")
def model_list() -> None:
    """List downloaded models."""
    cfg = config.load()
    _refuse_ollama_only(cfg, "there is no Ollama model list to show.")
    installed = engine.list_models(
        cfg.engine_url, key=cfg.engine_key, managed=cfg.engine_managed, sys=system.REAL
    )
    if not installed:
        ui.REAL.say("No models yet — run `lepika model add`.", markup=False)
        return
    table = Table(title="models")
    table.add_column("Name")
    table.add_column("Size", justify="right")
    for name, size in installed:
        marker = " [dim](default)[/dim]" if cfg.model and engine.same_model(name, cfg.model) else ""
        table.add_row(escape(name) + marker, engine.human_size(size))
    ui.REAL.say(table)


@model_app.command("rm")
def model_rm(
    name: str = typer.Argument(..., help="Model name as shown by `lepika model list`."),
) -> None:
    """Remove a downloaded model."""
    cfg = config.load()
    _refuse_ollama_only(cfg, "there is nothing to remove from Ollama.")
    engine.delete_model(
        cfg.engine_url, name, key=cfg.engine_key, managed=cfg.engine_managed, sys=system.REAL
    )
    ui.REAL.say(f"[green]✓ Removed:[/green] {escape(name)}")
    if cfg.model and engine.same_model(name, cfg.model):
        # A config still naming a model the engine no longer has makes `status`
        # and `up` claim a default that cannot answer. `_ready` says "No model
        # yet" once the field is empty.
        cfg.model = ""
        config.save(cfg)
        ui.REAL.say("That was the default model — run `lepika model add` to pick another.")


def _utf8_stdio(streams: Iterable[TextIO] | None = None) -> None:
    """Make stdout/stderr UTF-8 so ✓ and ✗ can be written wherever they go.

    Windows opens a piped or redirected stdout with the ANSI code page, which has
    no ✓ — every status line would then die with a UnicodeEncodeError traceback.
    The wrapper is reconfigured in place, not replaced, so Rich and Typer, which
    read sys.stdout at write time, follow along.
    """
    for stream in streams if streams is not None else (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and stream.encoding.lower() not in (
            "utf-8",
            "utf8",
        ):
            stream.reconfigure(encoding="utf-8")


def _fail(problem: str, fix: str) -> None:
    ui.REAL.say_error(f"[red]✗ {escape(problem)}[/red]")
    ui.REAL.say_error(f"[yellow]→ {escape(fix)}[/yellow]")


def _unexpected(exc: Exception) -> NoReturn:
    """A bug: the traceback goes to the log, the user gets where to find it."""
    fix = f"Please report it: {ISSUES_URL}"
    # The log itself may be the broken part (a read-only home, a full disk, no home
    # directory at all): then the user still gets the two lines, without a file to
    # look in. The path lookup sits inside the guard for the same reason.
    with contextlib.suppress(Exception):
        log.get_logger().error("cli.crash", error=repr(exc), traceback=traceback.format_exc())
        fix = f"Details are in {paths.home_path() / 'logs' / log.LOG_FILE}. {fix}"
    _fail(f"LePika hit an unexpected error: {type(exc).__name__}: {exc}", fix)
    raise SystemExit(1) from exc


def run() -> None:
    """Console-script entry point.

    The last line of defence for rule 2: whatever escapes a command becomes two
    lines and an exit code, never a traceback. `typer.Exit` never reaches the
    handlers: Click's standalone mode turns it into `SystemExit`, a BaseException
    that passes through untouched.
    """
    _utf8_stdio()
    try:
        app()
    except FriendlyError as exc:
        _fail(exc.problem, exc.fix)
        raise SystemExit(1) from exc
    except OSError as exc:
        # Only an error that really describes a file: an HTTPError is an OSError too,
        # with a URL for a filename and no strerror.
        if exc.strerror is None or not isinstance(exc.filename, (str, bytes, os.PathLike)):
            _unexpected(exc)
        # A read or a write, in LePika's home or the user's own folder (`model
        # import <dir>`): name the file and the folder it is in, nothing assumed.
        path = os.fsdecode(exc.filename)
        if (
            isinstance(exc, FileNotFoundError)
            and not os.path.dirname(path)
            and not os.path.lexists(path)
        ):
            # A bare name that is nowhere on disk is a program looked up on PATH: one
            # that is not installed. A bare name that exists — a dangling link in the
            # folder `model import .` reads — is a file.
            missing = CommandNotFound(path)
            _fail(missing.problem, missing.fix)
            raise SystemExit(1) from exc
        _fail(
            f"Could not access {path}: {exc.strerror}",
            f"Check the permissions on {Path(path).parent} and that the disk is not full.",
        )
        raise SystemExit(1) from exc
    except Exception as exc:
        _unexpected(exc)
