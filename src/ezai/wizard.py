"""The default `ezai` experience: detect, ask, install, open browser."""

from __future__ import annotations

from collections.abc import Callable

from rich.console import Console
from rich.markup import escape
from rich.prompt import Prompt
from rich.table import Table

from ezai import config, detect, express, models
from ezai.detect import SystemInfo
from ezai.errors import FriendlyError
from ezai.models import CuratedModel, ModelRef

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
    table = Table(title=f"Models that fit your {info.ram_gb:.0f} GB")
    table.add_column("#")
    table.add_column("Model")
    table.add_column("Ref")
    for i, m in enumerate(fitting, start=1):
        # Curated entries can come from the remote list: escape before rendering.
        table.add_row(str(i), escape(m.name), escape(m.ref))
    console.print(table)
    answer = ask_fn("Pick a number, or type any model (qwen3:8b · hf.co/<org>/<repo>-GGUF)").strip()
    # isdecimal, not isdigit: "²".isdigit() is True but int("²") raises.
    if answer.isdecimal() and 1 <= int(answer) <= len(fitting):
        return _validate(models.parse_model_ref(fitting[int(answer) - 1].ref))
    return _validate(models.parse_model_ref(answer))


def run_wizard(dry_run: bool = False) -> None:
    info = detect.detect()
    console.print(detect.plan_sentence(info))
    ref = choose_model(info)
    cfg = config.load()
    cfg.model = ref.raw
    config.save(cfg)
    if dry_run:
        console.print("would: ensure Ollama is installed and running")
        console.print(f"would: pull model {escape(ref.raw)}")
        console.print(f"would: start OpenWebUI on port {cfg.webui_port}")
        console.print(f"would: open {express.webui_url(cfg.webui_port)}")
        return
    express.ensure_ollama(info)
    express.pull_model(ref)
    express.ensure_openwebui(cfg)
    url = express.webui_url(cfg.webui_port)
    console.print(f"[green]✓ Ready:[/green] {url}")
    from ezai import cli

    cli._open_browser(url)
