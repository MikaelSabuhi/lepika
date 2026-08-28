"""The default `lepika` experience: detect, ask, install, open browser."""

from __future__ import annotations

from collections.abc import Callable

from rich.console import Console
from rich.markup import escape
from rich.prompt import Prompt
from rich.table import Table

from lepika import config, detect, engine, express, models
from lepika.detect import SystemInfo
from lepika.errors import FriendlyError
from lepika.models import CuratedModel, ModelRef

AskFn = Callable[..., str]
console = Console()

_ask: AskFn = Prompt.ask


def _validate(ref: ModelRef) -> ModelRef:
    if ref.kind == "hf_repo":
        raise FriendlyError(
            "Full-weight Hugging Face repos need vLLM (Server mode on Linux + NVIDIA), "
            "which isn't available yet.",
            "Use a GGUF build instead, e.g. hf.co/<org>/<model>-GGUF",
        )
    return ref


def choose_model(
    info: SystemInfo,
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

    prompt = "Pick a number, or type any model (qwen3:8b · hf.co/<org>/<repo>-GGUF)"

    def picked(answer: str) -> ModelRef | None:
        # isdecimal, not isdigit: "²".isdigit() is True but int("²") raises.
        if answer.isdecimal() and 1 <= int(answer) <= len(fitting):
            return _validate(models.parse_model_ref(fitting[int(answer) - 1].ref))
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
    return _validate(models.parse_model_ref(answer))


def run_wizard(dry_run: bool = False) -> None:
    info = detect.detect()
    console.print(detect.plan_sentence(info))
    ref = choose_model(info)
    cfg = config.load()
    if dry_run:
        cfg.model = ref.raw
        config.save(cfg)
        console.print("would: ensure Ollama is installed and running")
        console.print(f"would: pull model {escape(ref.raw)}")
        console.print(f"would: start OpenWebUI on port {cfg.webui_port}")
        console.print(f"would: open {express.webui_url(cfg.webui_port)}")
        return

    def pull_then_save() -> None:
        # Saved only once the pull succeeded: recording a model the machine failed
        # to download leaves the config pointing at something that isn't there.
        engine.pull_model(cfg.engine_url, ref, key=cfg.engine_key)
        cfg.model = ref.raw
        config.save(cfg)

    url = express.start_stack(info, cfg, after_engine=pull_then_save)
    from lepika import cli

    cli._ready(cfg, url)
