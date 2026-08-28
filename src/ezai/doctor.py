"""ezai doctor: every red ✗ comes with a one-line fix."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass

from ezai import config, detect, express
from ezai.detect import SystemInfo

# Advisory, not a hard failure: `cli.doctor` matches on this name, not a literal.
RAM_CHECK = "RAM"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    hint: str = ""


def run_checks(
    info: SystemInfo,
    which: Callable[[str], str | None] = shutil.which,
    api_up: Callable[..., bool] = detect.api_up,
    webui_up: Callable[..., bool] = express.webui_up,
) -> list[CheckResult]:
    cfg = config.load()
    return [
        CheckResult(
            "uv installed",
            which("uv") is not None,
            "Install uv: https://docs.astral.sh/uv/getting-started/installation/",
        ),
        CheckResult(
            "Ollama installed",
            info.has_ollama,
            "Run `ezai` to install it, or see https://ollama.com/download",
        ),
        CheckResult(
            "Ollama API responding",
            api_up(detect.OLLAMA_URL),
            "Run `ezai up` to start it; logs: `ezai logs`",
        ),
        CheckResult(
            "OpenWebUI responding",
            webui_up(cfg.webui_port),
            f"Run `ezai up`; if the port is busy, change webui_port in {config.config_path()}",
        ),
        CheckResult(
            RAM_CHECK,
            info.ram_gb >= 8.0,
            f"{info.ram_gb:.0f} GB detected — 8 GB+ recommended; stick to small "
            "models like qwen3:0.6b or llama3.2:3b",
        ),
    ]
