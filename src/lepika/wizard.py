"""The default `lepika` experience: detect, ask, install, open browser."""

from __future__ import annotations

import dataclasses

from rich.markup import escape
from rich.table import Table

from lepika import (
    acquire,
    config,
    detect,
    engine,
    express,
    gguf,
    health,
    hf,
    models,
    ollama,
    paths,
    server,
    serving,
    stack_env,
    system,
    ui,
)
from lepika.detect import Machine
from lepika.errors import FriendlyError
from lepika.models import CuratedModel, ModelRef
from lepika.system import System
from lepika.ui import UI, NoTerminal


def _validate(ref: ModelRef, cfg: config.Config, info: Machine) -> ModelRef:
    if ref.kind != "hf_repo":
        return ref
    if serving.vllm_allowed(cfg, info) or serving.import_allowed(cfg, info):
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


@dataclasses.dataclass(frozen=True)
class EngineMemory:
    """What the engine can load a model into besides a GPU, and how to say so."""

    gb: float
    docker: bool  # Docker Desktop's VM, smaller than the machine it runs on

    @property
    def label(self) -> str:
        return f"Docker's {self.gb:.0f} GB" if self.docker else f"{self.gb:.0f} GB RAM"

    @property
    def phrase(self) -> str:
        return self.label if self.docker else f"your {self.label}"


# Said wherever Docker's memory, not the machine's, is what kept a model out.
_MORE_FOR_DOCKER = "Docker Desktop → Settings → Resources gives it more."


def engine_memory(info: Machine, cfg: config.Config, *, sys: System) -> EngineMemory:
    """Server mode's engine is a container: on Docker Desktop it gets the VM's memory,
    not the machine's. Sized against the Mac's 16 GB, a 12 GB build looks fine and
    then never loads in the 7.75 GiB the VM really has. Anything else — Express, a
    remote engine, Linux where containers share the host's RAM, a Docker that will
    not answer — is sized against this machine.
    """
    if cfg.mode == "server" and cfg.engine_managed:
        docker = detect.docker_memory_gb(sys=sys)
        # 95 %: Linux reports the same kernel MemTotal; only a real VM is smaller.
        if 0 < docker < info.ram_gb * 0.95:
            return EngineMemory(docker, docker=True)
    return EngineMemory(info.ram_gb, docker=False)


def choose_quant(
    ref: ModelRef,
    cfg: config.Config,
    info: Machine,
    *,
    sys: System,
    ui: UI,
) -> ModelRef:
    """Turn a tagless `hf.co/<org>/<repo>` into `…:<TAG>` by asking, with a ★ default.

    A read, never a gate: whatever the Hub does — offline, gated, rate-limited — the
    ref goes on exactly as typed, so a pull that works today keeps working. An
    explicit tag, or any other ref shape, costs no network at all.
    """
    if ref.kind != "hf_gguf":
        return ref
    repo, tag = gguf.split_tag(ref.raw)
    if tag:
        return ref
    try:
        hf.check_repo(repo)
    except FriendlyError:
        return ref  # Ollama's own rejection says what is wrong with it
    try:
        builds = gguf.list_builds(repo, token=hf.token_for(cfg, sys=sys), sys=sys)
    except gguf.Unavailable:
        ui.say(
            f"[dim]Couldn't list the quantizations of {escape(repo)} — letting Ollama pick.[/dim]"
        )
        return ref
    if not builds:
        return ref  # nothing GGUF here: Ollama's refusal falls through to an import
    if cfg.mode == "server" and info.gpu == "apple":
        # A container never sees Metal: Server mode on a Mac is a CPU engine.
        gpu_gb = 0.0
    else:
        gpu_gb = detect.gpu_memory_gb(info, sys=sys)
    memory = engine_memory(info, cfg, sys=sys)
    more = f" {_MORE_FOR_DOCKER}" if memory.docker else ""
    shown = [b for b in builds if gguf.tier(b, gpu_gb, memory.gb) != "none"]
    if not shown:
        ui.say(
            f"None of the {len(builds)} builds fits in {memory.phrase} — letting Ollama pick."
            + more
        )
        return ref
    best = gguf.recommend(builds, gpu_gb, memory.gb)
    budget = f"{gpu_gb:.0f} GB GPU" if gpu_gb else memory.label
    table = Table(title=f"Quantizations of {escape(repo)}")
    table.add_column("#")
    table.add_column("Quant")
    table.add_column("Size", justify="right")
    table.add_column(f"Fit ({budget})")
    for i, b in enumerate(shown, start=1):
        # Quant names come from the Hub's file list: escape before rendering.
        name = escape(b.quant) + (" ★" if b == best else "")
        fit = gguf.label(gguf.tier(b, gpu_gb, memory.gb), gpu_gb)
        table.add_row(str(i), name, engine.human_size(b.size_bytes), fit)
    ui.say(table)
    hidden = len(builds) - len(shown)
    if hidden:
        plural = "s" if hidden != 1 else ""
        ui.say(
            f"[dim]{hidden} larger build{plural} hidden — too big for {memory.phrase}.{more}[/dim]"
        )
    if not cfg.engine_managed:
        ui.say(
            f"[dim]Sized against this machine — the engine at {escape(cfg.engine_url)} "
            "is what runs it.[/dim]"
        )

    default = str(shown.index(best) + 1) if best is not None else ""

    def chosen(quant: str) -> ModelRef:
        return ModelRef(raw=f"hf.co/{repo}:{quant}", kind="hf_gguf")

    for attempt in (1, 2):
        # `or default`: Prompt.ask already returns the default on an empty line, but a
        # `UI` reader may hand back "" — same guard as choose_mode.
        # No stdin to answer with — a script or CI — is not an error: the ★ is the
        # answer Enter would give, and the line below names it rather than its number.
        try:
            raw = ui.ask(
                "Pick a number",
                default=default or None,
                require_terminal=True,
            )
        except NoTerminal:
            if best is not None:
                ui.say(f"[dim]No terminal to ask — taking {escape(best.quant)}.[/dim]")
                return chosen(best.quant)
            ui.say("[dim]No terminal to ask — letting Ollama pick.[/dim]")
            return ref
        answer = raw.strip() or default
        if best is None and not answer:
            ui.say("[dim]Letting Ollama pick.[/dim]")
            return ref
        # isdecimal, not isdigit: "²".isdigit() is True but int("²") raises.
        if answer.isdecimal() and 1 <= int(answer) <= len(shown):
            return chosen(shown[int(answer) - 1].quant)
        if attempt == 1:
            ui.say(f"Pick 1 to {len(shown)}.")
    # Two misses: the recommendation is the answer that cannot be wrong — and with no
    # recommendation, the ref goes on as typed rather than looping.
    return chosen(best.quant) if best is not None else ref


def resolve(
    ref: ModelRef,
    cfg: config.Config,
    info: Machine,
    *,
    sys: System,
    ui: UI,
) -> ModelRef:
    """Validate a typed ref, then settle the one choice it may have left open."""
    return choose_quant(_validate(ref, cfg, info), cfg, info, sys=sys, ui=ui)


def choose_model(
    info: Machine,
    cfg: config.Config,
    curated: list[CuratedModel] | None = None,
    *,
    sys: System,
    ui: UI,
) -> ModelRef:
    candidates = curated if curated is not None else models.load_curated(sys=sys)
    memory = engine_memory(info, cfg, sys=sys)
    fitting = models.fitting(candidates, memory.gb)
    if not fitting:
        # A bare empty table reads as a broken program. Name the reason and the
        # way out — typing a ref still works, and something always fits.
        ui.say(
            f"Nothing in the curated list fits comfortably in {memory.phrase} — "
            "you can still type any model; try qwen3:0.6b"
        )
    else:
        table = Table(title=f"Models that fit {memory.phrase}")
        table.add_column("#")
        table.add_column("Model")
        table.add_column("Ref")
        for i, m in enumerate(fitting, start=1):
            # Curated entries can come from the remote list: escape before rendering.
            table.add_row(str(i), escape(m.name), escape(m.ref))
        ui.say(table)

    prompt = "Pick a number, or type any model (qwen3:8b · hf.co/<org>/<repo>-GGUF · <org>/<repo>)"

    def picked(answer: str) -> ModelRef | None:
        # isdecimal, not isdigit: "²".isdigit() is True but int("²") raises.
        if answer.isdecimal() and 1 <= int(answer) <= len(fitting):
            return _validate(models.parse_model_ref(fitting[int(answer) - 1].ref), cfg, info)
        return None

    def ask_model() -> str:
        try:
            return ui.ask(prompt).strip()
        except NoTerminal as exc:
            # There is no default model to fall back on — a guess would download
            # gigabytes nobody asked for.
            raise FriendlyError(
                "Choosing a model needs an answer, and there is no terminal to ask.",
                "Run `lepika` from an interactive terminal, or name the model: "
                "`lepika model add <model>`.",
            ) from exc

    answer = ask_model()
    chosen = picked(answer)
    if chosen is not None:
        return chosen
    if fitting and answer.isdecimal():
        # A number with no row behind it is a mistyped pick, not a model named 99.
        # One explanation, one retry — then it is taken at face value, so a wrong
        # second answer still ends the prompt rather than looping.
        ui.say(
            f"There are only {len(fitting)} numbered choices — pick 1 to "
            f"{len(fitting)}, or type a model name."
        )
        answer = ask_model()
        chosen = picked(answer)
        if chosen is not None:
            return chosen
    return resolve(models.parse_model_ref(answer), cfg, info, sys=sys, ui=ui)


def choose_mode(current: str, *, sys: System, ui: UI) -> str:
    """Express unless Docker is present AND the user picks Server.

    A machine without Docker never hears the question — and so is never nudged
    towards installing Docker for a mode it does not need.
    """
    if sys.which("docker") is None:
        return "express"
    default = "2" if current == "server" else "1"
    ui.say("How should LePika run?")
    ui.say("  1  ⚡ Express — native Ollama + OpenWebUI, GPU on every platform (recommended)")
    ui.say(
        "  2  🐳 Server — everything in docker compose; NVIDIA GPU on Linux, "
        "shareable with `lepika expose`"
    )
    choices = {"1": "express", "express": "express", "2": "server", "server": "server"}
    for attempt in (1, 2):
        # `or default`: Prompt.ask already returns the default on an empty line, but a
        # bare Enter must mean "keep what I have" whatever the prompt is.
        try:
            answer = ui.ask("Pick 1 or 2", default=default, require_terminal=True)
        except NoTerminal:
            # No terminal keeps the mode too — said by name, not as "1".
            kept = choices[default]
            ui.say(f"[dim]No terminal to ask — staying with {_MODE_LABEL[kept]}.[/dim]")
            return kept
        answer = answer.strip() or default
        if answer.lower() in choices:
            return choices[answer.lower()]
        if attempt == 1:
            ui.say("Pick 1 or 2 — or type express or server.")
    # Two misses: keep the mode the machine already has rather than loop.
    return choices[default]


_MODE_LABEL = {"express": "Express", "server": "Server"}


def _report_stop_failure(exc: FriendlyError, *, ui: UI) -> None:
    """Say a stop failed and carry on.

    A stack too broken to stop is usually why the user is leaving it, so this is a
    yellow line rather than a refusal that traps them in the mode that failed. Each
    stop reports its own; the problem sentence names which one it was.
    """
    ui.say(f"[yellow]Could not stop it cleanly: {escape(exc.problem)}[/yellow]")


def leave_mode(info: Machine, previous: str, cfg: config.Config, *, ui: UI) -> None:
    """Stop the stack we are switching away from, best effort.

    Both modes serve OpenWebUI on the same host port, so an abandoned stack does not
    just linger — it answers. `ensure_openwebui` would find the old container healthy,
    leave it alone, and open a UI that `lepika status` then reports as the other mode.
    """
    ui.say(
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
        backend.stop(info, old_cfg, sys=system.REAL)
    except FriendlyError as exc:
        _report_stop_failure(exc, ui=ui)
    # `lepika down` leaves Ollama up on purpose, but Server's pre-flight refuses a
    # port 11434 that a native engine holds. Only the one LePika started is stopped
    # — anything else has no pid file — and rule 9 keeps a remote engine untouched,
    # which is what `engine_managed` short-circuits before the call.
    if previous == "express" and old_cfg.engine_managed:
        try:
            if ollama.stop_ollama(
                info.os, old_cfg.engine_url, key=old_cfg.engine_key, sys=system.REAL
            ):
                ui.say("Stopped the Ollama LePika had started.")
        except FriendlyError as exc:
            _report_stop_failure(exc, ui=ui)


def run_wizard(dry_run: bool = False, mode: str | None = None) -> None:
    info = detect.detect(sys=system.REAL)
    cfg = config.load()
    previous = cfg.mode
    cfg.mode = mode if mode is not None else choose_mode(cfg.mode, sys=system.REAL, ui=ui.REAL)
    if cfg.mode == "server" and system.REAL.which("docker") is None:
        # Only reachable via `--mode server`: a clear error naming Express, never an
        # install prompt. Whether a present Docker is *running* is start_stack's
        # business — asking here would make --dry-run shell out.
        server.ensure_docker(sys=system.REAL)
    if cfg.mode != previous and not dry_run:
        leave_mode(info, previous, cfg, ui=ui.REAL)
    ui.REAL.say(detect.plan_sentence(info, cfg.mode, engine=serving.engine_label(cfg)))
    # The quant picker may read the Hub and ask nvidia-smi here, dry run or not: both are
    # reads, and the dry-run report then names the tag it would pull.
    ref = choose_model(info, cfg, sys=system.REAL, ui=ui.REAL)
    if dry_run:
        # Nothing is written: a dry run that saved the config would leave the machine
        # in a state the run said it would only describe.
        if cfg.mode == "server":
            # Built from lepika_home(), not stack_dir(): a dry run creates nothing.
            env_path = paths.lepika_home() / "stack" / stack_env.ENV_FILE
            ui.REAL.say(f"would: write {escape(str(env_path))}")
            ui.REAL.say("would: run docker compose up -d")
        else:
            ui.REAL.say("would: ensure Ollama is installed and running")
        if models.uses_vllm(ref.raw) and serving.vllm_allowed(cfg, info):
            ui.REAL.say(f"would: start vLLM with {escape(ref.raw)}")
        elif ref.kind == "hf_repo":
            ui.REAL.say(f"would: import {escape(ref.raw)} into Ollama ({engine.IMPORT_QUANT})")
        else:
            ui.REAL.say(f"would: pull model {escape(ref.raw)}")
        ui.REAL.say(f"would: start OpenWebUI on port {cfg.webui_port}")
        ui.REAL.say(f"would: open {health.webui_url(cfg.webui_port)}")
        return
    if models.uses_vllm(ref.raw) and serving.vllm_allowed(cfg, info):
        server.hf_token_prompt(paths.stack_dir() / stack_env.ENV_FILE, sys=system.REAL, ui=ui.REAL)
        # compose interpolates VLLM_MODEL from the config, so this one has to be
        # saved before the stack starts rather than after. A start that then fails
        # is fixed by `lepika model add`, not by a stale config.
        cfg.model = ref.raw
        config.save(cfg)

    def pull_then_save() -> None:
        if models.uses_vllm(ref.raw) and serving.vllm_allowed(cfg, info):
            return  # vLLM downloads its own weights while starting; nothing to pull
        # The mode is saved here, not before `start_stack`: this hook runs only once
        # the engine is up, so a pre-flight that refuses can no longer leave a config
        # claiming a mode that never started.
        config.save(cfg)
        # The model is saved only once it is actually there: recording a model the
        # machine failed to download or import leaves the config pointing at
        # something that isn't.
        served = acquire.acquire(info, cfg, ref, sys=system.REAL, ui=ui.REAL)
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
            info,
            dataclasses.replace(cfg, model=ref.raw),
            after_engine=pull_then_save,
            sys=system.REAL,
            ui=ui.REAL,
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
                if server.stop(info, cfg, sys=system.REAL):
                    ui.REAL.say("Removed the containers the failed switch had created.")
            except FriendlyError as exc:
                _report_stop_failure(exc, ui=ui.REAL)
        raise
    cli._ready(cfg, url)
