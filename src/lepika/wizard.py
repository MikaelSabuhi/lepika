"""The default `lepika` experience: detect, ask, install, open browser."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from rich.console import Console
from rich.markup import escape
from rich.prompt import Prompt
from rich.table import Table

from lepika import acquire, config, detect, engine, express, models, paths, server
from lepika.detect import SystemInfo
from lepika.errors import FriendlyError
from lepika.models import CuratedModel, ModelRef

AskFn = Callable[..., str]
console = Console()

_ask: AskFn = Prompt.ask


def _validate(ref: ModelRef, cfg: config.Config, info: SystemInfo) -> ModelRef:
    if ref.kind != "hf_repo":
        return ref
    if server.vllm_allowed(cfg, info) or express.import_allowed(cfg, info):
        return ref
    if not cfg.engine_managed:
        raise FriendlyError(
            "Full-weight repos are imported on the engine's machine, and this engine is "
            "not LePika's.",
            "Run `lepika model add` there, or `lepika connect --local` first.",
        )
    raise FriendlyError(
        "Full-weight Hugging Face repos need Ollama's MLX engine (Apple Silicon, or an "
        "NVIDIA GPU on Linux/Windows) in Express mode — or vLLM in Server mode on "
        "Linux + NVIDIA.",
        "Use a GGUF build instead, e.g. hf.co/<org>/<repo>-GGUF.",
    )


def choose_model(
    info: SystemInfo,
    cfg: config.Config,
    ask: AskFn | None = None,
    curated: list[CuratedModel] | None = None,
) -> ModelRef:
    ask_fn: AskFn = ask if ask is not None else _ask
    candidates = curated if curated is not None else models.load_curated()
    fitting = models.fitting(candidates, info.ram_gb)
    if not fitting:
        # A bare empty table reads as a broken program. Name the reason and the
        # way out — typing a ref still works, and something always fits.
        console.print(
            f"Nothing in the curated list fits comfortably in {info.ram_gb:.0f} GB — "
            "you can still type any model; try qwen3:0.6b"
        )
    else:
        table = Table(title=f"Models that fit your {info.ram_gb:.0f} GB")
        table.add_column("#")
        table.add_column("Model")
        table.add_column("Ref")
        for i, m in enumerate(fitting, start=1):
            # Curated entries can come from the remote list: escape before rendering.
            table.add_row(str(i), escape(m.name), escape(m.ref))
        console.print(table)

    prompt = "Pick a number, or type any model (qwen3:8b · hf.co/<org>/<repo>-GGUF · <org>/<repo>)"

    def picked(answer: str) -> ModelRef | None:
        # isdecimal, not isdigit: "²".isdigit() is True but int("²") raises.
        if answer.isdecimal() and 1 <= int(answer) <= len(fitting):
            return _validate(models.parse_model_ref(fitting[int(answer) - 1].ref), cfg, info)
        return None

    answer = ask_fn(prompt).strip()
    chosen = picked(answer)
    if chosen is not None:
        return chosen
    if fitting and answer.isdecimal():
        # A number with no row behind it is a mistyped pick, not a model named 99.
        # One explanation, one retry — then it is taken at face value, so a wrong
        # second answer still ends the prompt rather than looping.
        console.print(
            f"There are only {len(fitting)} numbered choices — pick 1 to "
            f"{len(fitting)}, or type a model name."
        )
        answer = ask_fn(prompt).strip()
        chosen = picked(answer)
        if chosen is not None:
            return chosen
    return _validate(models.parse_model_ref(answer), cfg, info)


def choose_mode(info: SystemInfo, current: str, ask: AskFn | None = None) -> str:
    """Express unless Docker is present AND the user picks Server.

    A machine without Docker never hears the question — and so is never nudged
    towards installing Docker for a mode it does not need.
    """
    if not info.has_docker:
        return "express"
    ask_fn: AskFn = ask if ask is not None else _ask
    default = "2" if current == "server" else "1"
    console.print("How should LePika run?")
    console.print(
        "  1  ⚡ Express — native Ollama + OpenWebUI, GPU on every platform (recommended)"
    )
    console.print(
        "  2  🐳 Server — everything in docker compose; NVIDIA GPU on Linux, "
        "shareable with `lepika expose`"
    )
    # `or default`: Prompt.ask already returns the default on an empty line, but a
    # bare Enter must mean "keep what I have" whatever the prompt is.
    answer = ask_fn("Pick 1 or 2", default=default).strip() or default
    return "server" if answer == "2" else "express"


_MODE_LABEL = {"express": "Express", "server": "Server"}


def _report_stop_failure(exc: FriendlyError) -> None:
    """Say a stop failed and carry on.

    A stack too broken to stop is usually why the user is leaving it, so this is a
    yellow line rather than a refusal that traps them in the mode that failed. Each
    stop reports its own; the problem sentence names which one it was.
    """
    console.print(f"[yellow]Could not stop it cleanly: {escape(exc.problem)}[/yellow]")


def leave_mode(info: SystemInfo, previous: str, cfg: config.Config) -> SystemInfo:
    """Stop the stack we are switching away from, best effort.

    Both modes serve OpenWebUI on the same host port, so an abandoned stack does not
    just linger — it answers. `ensure_openwebui` would find the old container healthy,
    leave it alone, and open a UI that `lepika status` then reports as the other mode.

    Returns `info` as it stands after the stop: detection ran before it, and Server's
    pre-flight would otherwise refuse port 11434 for an Ollama that is already gone.
    """
    console.print(
        f"Switching from {_MODE_LABEL[previous]} to {_MODE_LABEL[cfg.mode]} mode — "
        f"stopping the {_MODE_LABEL[previous]} stack…"
    )
    # The config as it was loaded: `stop` reads the old mode's ports, not the new mode's.
    old_cfg = dataclasses.replace(cfg, mode=previous)
    backend = server if previous == "server" else express
    # A try each, not one around both: a UI that will not die would otherwise take
    # the engine stop down with it, and the port 11434 the next mode needs stays held.
    try:
        # The bool is "was anything running", which is not a reason to stop switching.
        backend.stop(info, old_cfg)
    except FriendlyError as exc:
        _report_stop_failure(exc)
    # `lepika down` leaves Ollama up on purpose, but Server's pre-flight refuses a
    # port 11434 that a native engine holds. Only the one LePika started is stopped
    # — anything else has no pid file — and rule 9 keeps a remote engine untouched,
    # which is what `engine_managed` short-circuits before the call.
    if previous == "express" and old_cfg.engine_managed:
        try:
            if express.stop_ollama(info.os, old_cfg.engine_url, key=old_cfg.engine_key):
                console.print("Stopped the Ollama LePika had started.")
                # True means the API went quiet, not just that a signal was sent: the
                # snapshot is corrected from what was verified, never re-probed.
                info = dataclasses.replace(info, ollama_running=False)
        except FriendlyError as exc:
            _report_stop_failure(exc)
    return info


def run_wizard(dry_run: bool = False, mode: str | None = None) -> None:
    info = detect.detect()
    cfg = config.load()
    previous = cfg.mode
    cfg.mode = mode if mode is not None else choose_mode(info, cfg.mode)
    if cfg.mode == "server" and not info.has_docker:
        # Only reachable via `--mode server`: a clear error naming Express, never an
        # install prompt. Whether a present Docker is *running* is start_stack's
        # business — asking here would make --dry-run shell out.
        server.ensure_docker(info)
    if cfg.mode != previous and not dry_run:
        info = leave_mode(info, previous, cfg)
    console.print(detect.plan_sentence(info, cfg.mode, engine=server.engine_label(cfg)))
    ref = choose_model(info, cfg)
    if dry_run:
        # Nothing is written: a dry run that saved the config would leave the machine
        # in a state the run said it would only describe.
        if cfg.mode == "server":
            # Built from lepika_home(), not stack_dir(): a dry run creates nothing.
            env_path = paths.lepika_home() / "stack" / server.ENV_FILE
            console.print(f"would: write {escape(str(env_path))}")
            console.print("would: run docker compose up -d")
        else:
            console.print("would: ensure Ollama is installed and running")
        if models.uses_vllm(ref.raw) and server.vllm_allowed(cfg, info):
            console.print(f"would: start vLLM with {escape(ref.raw)}")
        elif ref.kind == "hf_repo":
            console.print(f"would: import {escape(ref.raw)} into Ollama ({engine.IMPORT_QUANT})")
        else:
            console.print(f"would: pull model {escape(ref.raw)}")
        console.print(f"would: start OpenWebUI on port {cfg.webui_port}")
        console.print(f"would: open {express.webui_url(cfg.webui_port)}")
        return
    if models.uses_vllm(ref.raw) and server.vllm_allowed(cfg, info):
        server.hf_token_prompt(paths.stack_dir() / server.ENV_FILE)
        # compose interpolates VLLM_MODEL from the config, so this one has to be
        # saved before the stack starts rather than after. A start that then fails
        # is fixed by `lepika model add`, not by a stale config.
        cfg.model = ref.raw
        config.save(cfg)

    def pull_then_save() -> None:
        if models.uses_vllm(ref.raw) and server.vllm_allowed(cfg, info):
            return  # vLLM downloads its own weights while starting; nothing to pull
        # The mode is saved here, not before `start_stack`: this hook runs only once
        # the engine is up, so a pre-flight that refuses can no longer leave a config
        # claiming a mode that never started.
        config.save(cfg)
        # The model is saved only once it is actually there: recording a model the
        # machine failed to download or import leaves the config pointing at
        # something that isn't.
        served = acquire.acquire(info, cfg, ref)
        if served is None:
            return  # declined: the stack is up, the model is chosen next time
        cfg.model = served
        config.save(cfg)

    from lepika import cli

    # The stack is started for the model about to be pulled, not the one still saved:
    # in Server mode the profile follows the ref, so an old full-weight repo would
    # bring up vLLM and leave nothing to pull into. Only the pull saves it (above).
    try:
        url = cli._backend(cfg).start_stack(
            info, dataclasses.replace(cfg, model=ref.raw), after_engine=pull_then_save
        )
    except FriendlyError:
        # A switch into Server that fails before the engine came up never saved the
        # config, so the containers `compose up` created are invisible: `lepika down`
        # (still the old mode) says "Nothing was running" while they hold the ports
        # and answer readiness checks for the next start. Sweep them before leaving.
        # Once the on-disk mode says server they are visible, and a re-run failure
        # must not tear down what was already running — hence the reload.
        if cfg.mode == "server" and previous != "server" and config.load().mode != "server":
            try:
                if server.stop(info, cfg):
                    console.print("Removed the containers the failed switch had created.")
            except FriendlyError as exc:
                _report_stop_failure(exc)
        raise
    cli._ready(cfg, url)
