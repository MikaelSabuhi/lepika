"""Typer entry point for LePika."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.metadata
import io
import os
import sys as _sys
import traceback
import webbrowser
from collections.abc import Callable, Iterable
from dataclasses import dataclass
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
    paths,
    server,
    serving,
    stack_env,
    system,
    ui,
    wizard,
)
from lepika.backend import Backend
from lepika.detect import Machine
from lepika.errors import ISSUES_URL, CommandNotFound, FriendlyError
from lepika.system import System
from lepika.ui import UI
from lepika.updates import UpdateStep

app = typer.Typer(
    help="One command → local AI chat in your browser.",
    add_completion=False,
)


@dataclass(frozen=True)
class Deps:
    """What a command works through. Built once per run; tests hand in their own."""

    sys: System
    ui: UI
    machine: Callable[[], Machine]  # detect this machine, when a command needs it
    backends: Callable[[config.Config], Backend]  # the backend for a config's mode

    @classmethod
    def real(cls) -> Deps:
        # The lambdas read the module globals when they are called, not now: the test
        # suite replaces these globals, sometimes after this has already run.
        return cls(
            sys=system.REAL,
            ui=ui.REAL,
            machine=lambda: detect.detect(sys=system.REAL),
            backends=lambda cfg: backend.backend_for(cfg, sys=system.REAL, ui=ui.REAL),
        )


def _deps(ctx: typer.Context) -> Deps:
    """The `Deps` the root callback put on the context (or a caller handed in)."""
    if not isinstance(ctx.obj, Deps):
        raise TypeError(f"the command context holds {ctx.obj!r}, not Deps")
    return ctx.obj


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
    # Click runs this before any subcommand, whose context inherits the object.
    if ctx.obj is None:
        ctx.obj = Deps.real()
    deps = _deps(ctx)
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
        result = wizard.run_wizard(
            deps.machine(),
            dry_run=dry_run,
            mode=mode,
            sys=deps.sys,
            ui=deps.ui,
            backends=deps.backends,
        )
        if result is not None:
            _ready(*result, ui=deps.ui)


def _open_browser(url: str) -> None:
    webbrowser.open(url)


def _ready(cfg: config.Config, url: str, *, ui: UI) -> None:
    """Announce a running stack and open it — shared by `lepika up` and the wizard."""
    ui.say(f"[green]✓ Ready:[/green] {url}")
    if not cfg.model:
        # A chat UI with no model behind it looks broken; say what's missing.
        ui.say("No model yet — run `lepika` or `lepika model add`.")
    _open_browser(url)


@app.command()
def up(ctx: typer.Context) -> None:
    """Start the local AI stack and open the browser."""
    deps = _deps(ctx)
    info = deps.machine()
    cfg = config.load()
    deps.ui.say(detect.plan_sentence(info, cfg.mode, engine=serving.engine_label(cfg)))
    chosen = deps.backends(cfg)
    # The notes are asked before the start, and that order is the point: Express asks
    # whether an engine already answers, and one that does is what the start adopts,
    # so config.toml's context length cannot reach it.
    for note in chosen.notes(info, cfg):
        deps.ui.say(f"[yellow]{escape(note)}[/yellow]")
    _ready(cfg, chosen.start(info, cfg), ui=deps.ui)


@app.command()
def down(ctx: typer.Context) -> None:
    """Stop the stack (Express: OpenWebUI only; Server: every container, models kept)."""
    deps = _deps(ctx)
    cfg = config.load()
    if deps.backends(cfg).stop(deps.machine(), cfg):
        deps.ui.say("[green]✓ Stopped.[/green]")
    else:
        deps.ui.say("Nothing was running.")


@app.command()
def status(ctx: typer.Context) -> None:
    """Show what's running."""
    deps = _deps(ctx)
    cfg = config.load()
    table = Table(title="lepika status")
    table.add_column("Service")
    table.add_column("State")
    # First row: every other row means something different in each mode.
    table.add_row("Mode", cfg.mode)
    vllm = serving.vllm_active(cfg)
    engine_ok = (
        health.vllm_up(server.VLLM_URL, sys=deps.sys)
        if vllm
        else health.api_up(cfg.engine_url, key=cfg.engine_key, sys=deps.sys)
    )
    webui_ok = health.webui_up(cfg.webui_port, sys=deps.sys)
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
    deps.ui.say(table)


@app.command()
def logs(
    ctx: typer.Context, lines: int = typer.Option(50, min=1, help="Lines per log file.")
) -> None:
    """Print the tail of LePika's logs (Server mode: the container logs too)."""
    deps = _deps(ctx)
    sections = deps.backends(config.load()).logs(lines)
    if not sections:
        # Silence is indistinguishable from a broken command.
        deps.ui.say("(no logs yet)")
        return
    for title, text in sections:
        deps.ui.say(Rule(title))
        deps.ui.say(text, markup=False)


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Diagnose the local setup."""
    # Imported here, not at module scope: this command function shadows the name.
    from lepika import doctor as doctor_mod

    deps = _deps(ctx)
    info = deps.machine()
    results = doctor_mod.run_checks(info, sys=deps.sys)
    core_failed = False
    for r in results:
        mark = "[green]✓[/green]" if r.ok else "[red]✗[/red]"
        deps.ui.say(f"{mark} {r.name}")
        if not r.ok:
            deps.ui.say(f"  [yellow]→ {escape(r.hint)}[/yellow]")
            if r.name != doctor_mod.RAM_CHECK:
                core_failed = True
    if core_failed:
        raise typer.Exit(code=1)


@app.command()
def update(ctx: typer.Context) -> None:
    """Upgrade the engine and OpenWebUI to their latest versions."""
    deps = _deps(ctx)
    deps.ui.say("Upgrading…")
    cfg = config.load()
    steps: list[UpdateStep] = deps.backends(cfg).update(deps.machine(), cfg)
    for step in steps:
        detail = escape(step.detail)
        if step.outcome in ("stale", "note"):
            deps.ui.say(f"[yellow]! {step.component}: {detail}[/yellow]")
            continue
        suffix = f" ({detail})" if detail else ""
        verb = "upgraded" if step.outcome == "upgraded" else "up to date"
        deps.ui.say(f"[green]✓ {step.component}: {verb}{suffix}[/green]")


@app.command()
def connect(
    ctx: typer.Context,
    url: str | None = typer.Argument(None, help="Engine URL, e.g. http://gpu-box:11435"),
    key: str = typer.Option("", "--key", help="API key, if the engine needs one."),
    local: bool = typer.Option(False, "--local", help="Go back to the engine on this machine."),
) -> None:
    """Use an engine running on another machine (or --local to stop doing so)."""
    deps = _deps(ctx)
    cfg = config.load()
    if local:
        local_cfg = dataclasses.replace(
            cfg, engine_managed=True, engine_url=config.DEFAULT_ENGINE_URL, engine_key=""
        )

        def go_local() -> bool:
            log.get_logger().info("engine.connect", url=local_cfg.engine_url, local=True)
            return deps.backends(local_cfg).apply_engine_change(deps.machine(), local_cfg)

        # A withdrawal: saved first, so a failed switch still leaves the next
        # `lepika up` on the local engine.
        if config.save_before(local_cfg, go_local):
            deps.ui.say("[green]✓ Using the local engine again[/green] — OpenWebUI restarted.")
        else:
            deps.ui.say("[green]✓ Using the local engine again.[/green] Run `lepika up`.")
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
    if not health.api_up(url, key=key, sys=deps.sys):
        raise FriendlyError(
            f"No engine answered at {url}.",
            "Check the address and that `lepika expose` is on over there "
            "(add --key if it printed one).",
        )
    remote = dataclasses.replace(cfg, engine_managed=False, engine_url=url, engine_key=key)
    # Saved once the running stack uses it: a switch that fails must not leave the
    # config naming an engine nothing here is connected to.
    restarted = config.save_after(
        remote,
        lambda: deps.backends(remote).apply_engine_change(deps.machine(), remote),
    )
    log.get_logger().info("engine.connect", url=url, key=key)
    if restarted:
        deps.ui.say(f"[green]✓ Connected to[/green] {escape(url)} — OpenWebUI restarted.")
    else:
        deps.ui.say(f"[green]✓ Connected to[/green] {escape(url)}. Run `lepika up`.")


@app.command()
def expose(
    ctx: typer.Context,
    off: bool = typer.Option(False, "--off", help="Back to localhost only."),
    show: bool = typer.Option(False, "--show", help="Print the key and connect line again."),
    rotate: bool = typer.Option(False, "--rotate", help="Generate a new key."),
) -> None:
    """Share the engine (and the chat UI) with your network behind a generated API key."""
    deps = _deps(ctx)
    cfg = config.load()
    if cfg.mode != "server":
        raise FriendlyError(
            "`lepika expose` needs Server mode (it runs a small Caddy proxy in the stack).",
            "Run `lepika --mode server` on this machine first.",
        )
    if off:
        closed = dataclasses.replace(cfg, exposed=False)
        # A withdrawal: saved first, so a failed restart cannot have the next
        # `lepika up` put this machine back on the network.
        config.save_before(
            closed,
            lambda: deps.backends(closed).start(deps.machine(), closed),
        )
        log.get_logger().info("expose.off")
        deps.ui.say("[green]✓ Back to localhost only.[/green]")
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
        exposed = dataclasses.replace(cfg, exposed=True)
        # Saved once the stack is up with the proxy, never on the strength of a start
        # that failed.
        config.save_after(
            exposed,
            lambda: deps.backends(exposed).start(deps.machine(), exposed),
        )
        cfg = exposed
        log.get_logger().info("expose.on", api_port=cfg.api_port)
    ips = server.lan_ips(sys=deps.sys)
    # Nothing resolved at all (offline, or a locked-down host): say so in words
    # rather than print something that reads like an address.
    ip = ips[0] if ips else server.NO_ROUTE
    if cfg.exposed:
        deps.ui.say(f"[green]✓ Exposed.[/green] Chat UI: http://{ip}:{cfg.webui_port}")
    else:
        deps.ui.say("Not exposed yet — run `lepika expose` to turn it on.")
        deps.ui.say(f"Chat UI once it is on: http://{ip}:{cfg.webui_port}")
    if len(ips) > 1:
        # The route pick is the address this box would dial out on, which is not
        # always the one the laptop in the next room can reach.
        others = ", ".join(f"http://{other}:{cfg.webui_port}" for other in ips[1:])
        deps.ui.say(f"Also reachable on: {others}", markup=False, soft_wrap=True)
    deps.ui.say("The chat UI asks for a sign-in; the engine API wants the key below.")
    # soft_wrap on the copied lines: an 80-column wrap breaks a URL or a key.
    if serving.vllm_active(cfg):
        # `lepika connect` probes Ollama's /api/version, which Caddy in front of vLLM
        # never answers — so point at the OpenAI-compatible URL OpenWebUI can add.
        deps.ui.say(f"Engine API is vLLM (OpenAI-compatible): http://{ip}:{cfg.api_port}/v1")
        deps.ui.say(
            "On another machine, add that URL with the key as an OpenAI connection in "
            "OpenWebUI (Admin → Settings → Connections); `lepika connect` is for "
            "Ollama engines."
        )
        deps.ui.say(f"  Key: {key}", markup=False, soft_wrap=True)
        if rotate:
            # Engine-neutral: reconnecting here is editing the OpenAI connection in
            # OpenWebUI, and `lepika connect` refuses a vLLM engine outright.
            deps.ui.say("Machines using the old key need the new one above.")
    else:
        deps.ui.say(f"Engine API: http://{ip}:{cfg.api_port}")
        deps.ui.say("On another machine:")
        deps.ui.say(
            f"  lepika connect http://{ip}:{cfg.api_port} --key {key}", markup=False, soft_wrap=True
        )
        if rotate:
            # Caddy only honours the new key, so every machine holding the old one is
            # locked out until it is told — silence there looks like a broken engine.
            deps.ui.say(
                f"Machines that connected before need `lepika connect http://{ip}:{cfg.api_port} "
                f"--key {key}` again.",
                markup=False,
                soft_wrap=True,
            )
    deps.ui.say("Keep the key private; `lepika expose --rotate` replaces it.")


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
    ctx: typer.Context,
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
    deps = _deps(ctx)
    info = deps.machine()
    cfg = config.load()
    if ref is None:
        model_ref = wizard.choose_model(info, cfg, sys=deps.sys, ui=deps.ui)
    else:
        # Same rejection — and the same quant picker — as the wizard's, by reusing it.
        model_ref = wizard.resolve(models.parse_model_ref(ref), cfg, info, sys=deps.sys, ui=deps.ui)
    if models.uses_vllm(model_ref.raw) and serving.vllm_allowed(cfg, info):
        # Nothing to pull: vLLM downloads the weights itself when it starts, from the
        # model the start writes into the .env that compose interpolates.
        server.hf_token_prompt(paths.stack_dir() / stack_env.ENV_FILE, sys=deps.sys, ui=deps.ui)
        with_model = dataclasses.replace(cfg, model=model_ref.raw)

        def compose_up_succeeded() -> None:
            # Saved before vLLM's readiness wait, which can outlast a half-hour download:
            # a timeout there must not unsave a model the next `lepika up` keeps fetching.
            config.save(with_model)
            deps.ui.say(f"[green]✓ Set:[/green] {escape(model_ref.raw)} — starting vLLM…")

        _ready(
            with_model,
            deps.backends(with_model).start(info, with_model, after_engine=compose_up_succeeded),
            ui=deps.ui,
        )
        return
    # The engine the *new* model needs: in Server mode the profile follows the ref, and
    # switching away from a full-weight repo with the old ref still in the config
    # would bring up vLLM, leave ollama stopped, and fail the pull.
    deps.backends(cfg).ensure_engine(info, dataclasses.replace(cfg, model=model_ref.raw))
    cfg, served = acquire.acquire(info, cfg, model_ref, quant=quant, sys=deps.sys, ui=deps.ui)
    if served is None:
        deps.ui.say("Nothing added.")
        return
    cfg = dataclasses.replace(cfg, model=served)
    config.save(cfg)
    deps.ui.say(f"[green]✓ Added:[/green] {escape(served)}")


@model_app.command("import")
def model_import(
    ctx: typer.Context,
    path: Path = WEIGHTS_ARGUMENT,
    name: str = typer.Option("", "--name", help="What Ollama calls it (default: the folder name)."),
    quant: str = QUANT_OPTION,
) -> None:
    """Import safetensors weights already on disk into Ollama."""
    quant = _checked_quant(quant)
    deps = _deps(ctx)
    info = deps.machine()
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
    ollama.ensure_ollama(info, url=cfg.engine_url, context_length=cfg.context_length, sys=deps.sys)
    # Resolved from here on: the "Importing … from …" line and the `source=` log field
    # name the folder unambiguously, whatever the shell was sitting in.
    served = acquire.import_local(info, cfg, path.resolve(), name, quant, sys=deps.sys, ui=deps.ui)
    cfg = dataclasses.replace(cfg, model=served)
    config.save(cfg)
    deps.ui.say(f"[green]✓ Imported:[/green] {escape(served)}")


@model_app.command("list")
def model_list(ctx: typer.Context) -> None:
    """List downloaded models."""
    deps = _deps(ctx)
    cfg = config.load()
    _refuse_ollama_only(cfg, "there is no Ollama model list to show.")
    installed = engine.list_models(
        cfg.engine_url, key=cfg.engine_key, managed=cfg.engine_managed, sys=deps.sys
    )
    if not installed:
        deps.ui.say("No models yet — run `lepika model add`.", markup=False)
        return
    table = Table(title="models")
    table.add_column("Name")
    table.add_column("Size", justify="right")
    for name, size in installed:
        marker = " [dim](default)[/dim]" if cfg.model and engine.same_model(name, cfg.model) else ""
        table.add_row(escape(name) + marker, engine.human_size(size))
    deps.ui.say(table)


@model_app.command("rm")
def model_rm(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Model name as shown by `lepika model list`."),
) -> None:
    """Remove a downloaded model."""
    deps = _deps(ctx)
    cfg = config.load()
    _refuse_ollama_only(cfg, "there is nothing to remove from Ollama.")
    engine.delete_model(
        cfg.engine_url, name, key=cfg.engine_key, managed=cfg.engine_managed, sys=deps.sys
    )
    deps.ui.say(f"[green]✓ Removed:[/green] {escape(name)}")
    if cfg.model and engine.same_model(name, cfg.model):
        # A config still naming a model the engine no longer has makes `status`
        # and `up` claim a default that cannot answer. `_ready` says "No model
        # yet" once the field is empty.
        cfg = dataclasses.replace(cfg, model="")
        config.save(cfg)
        deps.ui.say("That was the default model — run `lepika model add` to pick another.")


def _utf8_stdio(streams: Iterable[TextIO] | None = None) -> None:
    """Make stdout/stderr UTF-8 so ✓ and ✗ can be written wherever they go.

    Windows opens a piped or redirected stdout with the ANSI code page, which has
    no ✓ — every status line would then die with a UnicodeEncodeError traceback.
    The wrapper is reconfigured in place, not replaced, so Rich and Typer, which
    read sys.stdout at write time, follow along.
    """
    for stream in streams if streams is not None else (_sys.stdout, _sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and stream.encoding.lower() not in (
            "utf-8",
            "utf8",
        ):
            stream.reconfigure(encoding="utf-8")


def _fail(problem: str, fix: str, *, ui: UI) -> None:
    ui.say_error(f"[red]✗ {escape(problem)}[/red]")
    ui.say_error(f"[yellow]→ {escape(fix)}[/yellow]")


def _unexpected(exc: Exception, *, ui: UI) -> NoReturn:
    """A bug: the traceback goes to the log, the user gets where to find it."""
    fix = f"Please report it: {ISSUES_URL}"
    # The log itself may be the broken part (a read-only home, a full disk, no home
    # directory at all): then the user still gets the two lines, without a file to
    # look in. The path lookup sits inside the guard for the same reason.
    with contextlib.suppress(Exception):
        log.get_logger().error("cli.crash", error=repr(exc), traceback=traceback.format_exc())
        fix = f"Details are in {paths.home_path() / 'logs' / log.LOG_FILE}. {fix}"
    _fail(f"LePika hit an unexpected error: {type(exc).__name__}: {exc}", fix, ui=ui)
    raise SystemExit(1) from exc


def run() -> None:
    """Console-script entry point.

    The last line of defence for rule 2: whatever escapes a command becomes two
    lines and an exit code, never a traceback. `typer.Exit` never reaches the
    handlers: Click's standalone mode turns it into `SystemExit`, a BaseException
    that passes through untouched.
    """
    _utf8_stdio()
    # Outside the guard: building it only reads two module globals and makes two
    # closures, and the handlers below print through its UI.
    deps = Deps.real()
    try:
        app(obj=deps)
    except FriendlyError as exc:
        _fail(exc.problem, exc.fix, ui=deps.ui)
        raise SystemExit(1) from exc
    except OSError as exc:
        # Only an error that really describes a file: an HTTPError is an OSError too,
        # with a URL for a filename and no strerror.
        if exc.strerror is None or not isinstance(exc.filename, (str, bytes, os.PathLike)):
            _unexpected(exc, ui=deps.ui)
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
            _fail(missing.problem, missing.fix, ui=deps.ui)
            raise SystemExit(1) from exc
        _fail(
            f"Could not access {path}: {exc.strerror}",
            f"Check the permissions on {Path(path).parent} and that the disk is not full.",
            ui=deps.ui,
        )
        raise SystemExit(1) from exc
    except Exception as exc:
        _unexpected(exc, ui=deps.ui)
