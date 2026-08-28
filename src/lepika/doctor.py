"""lepika doctor: every red ✗ comes with a one-line fix."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass

from lepika import config, detect, express
from lepika.detect import SystemInfo

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
    checks = [
        CheckResult(
            "uv installed",
            which("uv") is not None,
            "Install uv: https://docs.astral.sh/uv/getting-started/installation/",
        ),
    ]
    if cfg.engine_managed:
        # Only asked when the engine is ours to install: a remote one is not.
        checks.append(
            CheckResult(
                "Ollama installed",
                info.has_ollama,
                "Run `lepika` to install it, or see https://ollama.com/download",
            )
        )
    checks += [
        CheckResult(
            "Engine responding",
            api_up(cfg.engine_url, key=cfg.engine_key),
            "Run `lepika up` to start it; logs: `lepika logs`"
            if cfg.engine_managed
            else f"{cfg.engine_url} is not answering — check that machine, "
            "or `lepika connect --local`",
        ),
        CheckResult(
            "OpenWebUI responding",
            webui_up(cfg.webui_port),
            f"Run `lepika up`; if the port is busy, change webui_port in {config.config_path()}",
        ),
        CheckResult(
            RAM_CHECK,
            info.ram_gb >= 8.0,
            f"{info.ram_gb:.0f} GB detected — 8 GB+ recommended; stick to small "
            "models like qwen3:0.6b or llama3.2:3b",
        ),
    ]
    return checks
