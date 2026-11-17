"""Pull or import: the one place that decides how a model ref gets into Ollama.

`model add` and the wizard both come here, so the policy in rule 10 — Ollama
decides an `hf.co/…` ref, the Hugging Face file list decides a bare `org/repo` —
is written once, along with the sizing and the single interactive question.
`model import` lands here too: weights already on disk skip the download, but
they get the same name, disk and RAM checks before Ollama is asked to build.
"""

from __future__ import annotations

import contextlib
import dataclasses
import re
import shutil
from pathlib import Path

from rich.markup import escape

from lepika import config, engine, gguf, hf, mlx, models, paths, serving
from lepika.detect import Machine
from lepika.errors import FriendlyError
from lepika.models import ModelRef
from lepika.system import System
from lepika.ui import UI, NoTerminal


def _confirm(question: str, *, ui: UI) -> bool:
    """The one interactive question on the import path."""
    try:
        # Enter means no, as it always has; no terminal is not taken as a no, because
        # the caller would then report a decline nobody made.
        return ui.confirm(question, default=False, require_terminal=True)
    except NoTerminal as exc:
        # Piped, redirected or in CI: rich sees EOF on stdin. A download this size is
        # never started on a guess, so say what is missing instead of tracebacking.
        raise FriendlyError(
            "Importing a full-weight repo needs a yes/no answer, and there is no terminal to ask.",
            "Run `lepika model add <repo>` from an interactive shell.",
        ) from exc


def preflight_with_token(
    repo: str, cfg: config.Config, *, sys: System, ui: UI
) -> tuple[hf.Preflight, str, config.Config]:
    """List the repo; on a gated answer ask for a token once and try again.

    Also the config as it now is on disk. A typed token is a credential, saved the
    moment it is given (0600, like every config write), so a download that fails
    later never makes the user type it again.
    """
    token = hf.token_for(cfg, sys=sys)
    try:
        return hf.preflight(repo, token, sys=sys), token, cfg
    except hf.GatedRepo:
        token = hf.ask_token(ui=ui)
        if not token:
            raise
        cfg = dataclasses.replace(cfg, hf_token=token)
        config.save(cfg)
        return hf.preflight(repo, token, sys=sys), token, cfg


def _ram_advisory(label: str, quantized: int, info: Machine, *, ui: UI) -> None:
    """Advice, never a refusal: it is the user's machine and the user's model."""
    if quantized > info.ram_gb * 0.75 * 2**30:
        ui.say(
            f"[yellow]{escape(label)} is about {engine.human_size(quantized)} after "
            f"quantization — it may not fit comfortably in {info.ram_gb:.0f} GB.[/yellow]"
        )


def import_repo(
    info: Machine,
    cfg: config.Config,
    ref: ModelRef,
    quant: str = engine.IMPORT_QUANT,
    *,
    sys: System,
    ui: UI,
) -> tuple[config.Config, str | None]:
    """Download a full-weight repo and import it into Ollama.

    The config as it now is on disk (a typed token is saved on the way), and the name
    Ollama now serves — None when the user declined the download.
    """
    repo = ref.raw
    installed = engine.list_models(
        cfg.engine_url, key=cfg.engine_key, managed=cfg.engine_managed, sys=sys
    )
    if any(engine.same_model(name, repo) for name, _size in installed):
        # Ollama already serves it: downloading the weights again to rebuild the same
        # model is minutes and gigabytes for nothing.
        ui.say(f"{escape(repo)} is already imported.")
        return cfg, repo
    pre, token, cfg = preflight_with_token(repo, cfg, sys=sys, ui=ui)
    if pre.has_gguf and not pre.has_safetensors:
        # A GGUF build typed without `hf.co/`: Ollama pulls those itself.
        gguf = models.parse_model_ref(f"hf.co/{repo}")
        engine.pull_model(
            cfg.engine_url,
            gguf,
            key=cfg.engine_key,
            managed=cfg.engine_managed,
            sys=sys,
            ui=ui,
        )
        return cfg, gguf.raw
    if not pre.has_safetensors:
        raise FriendlyError(
            f"'{repo}' has no safetensors or GGUF weights.",
            "Pick a repo that ships model weights.",
        )
    # Already-quantized weights (NVFP4 and friends) import as-is: `-q` would make
    # Ollama refuse the whole build with "cannot requantize" — after the download.
    # The one small file that says so is fetched into the staging dir up front, so
    # the confirm question, the disk math and the RAM advisory all tell the truth.
    dest = hf.download_dir(repo)
    method = hf.quant_method(hf.fetch_config(repo, dest, token, sys=sys))
    # Only what download fetches (download_bytes skips the GGUF/PyTorch twins), plus
    # the copy Ollama writes — a quarter of a raw source, all of a quantized one.
    # Two filesystems can be involved: the download lands under ~/.lepika, while
    # Ollama writes into its own store — on a stock Linux service install that is
    # /usr/share/ollama/.ollama, often not the disk $HOME is on. The smaller of the
    # two decides, because either one filling up fails the import.
    store_bytes = pre.download_bytes if method is not None else pre.download_bytes // 4
    need = int((pre.download_bytes + store_bytes) * 1.1)
    disks = {paths.lepika_home()}  # created on access, so it is always there
    store = mlx.ollama_store(sys=sys)
    if store.exists():
        disks.add(store)
    free = min(sys.disk_free(disk) for disk in disks)
    if free < need:
        raise FriendlyError(
            f"Not enough disk: {repo} needs ~{engine.human_size(need)} free "
            f"({engine.human_size(free)} available).",
            "Free some space or pick a smaller model.",
        )
    _ram_advisory(repo, store_bytes, info, ui=ui)
    how = (
        f"import it as-is (already {escape(method)}-quantized, ~{engine.human_size(store_bytes)})"
        if method is not None
        else f"import as {quant} (~{engine.human_size(store_bytes)})"
    )
    if not _confirm(f"Fetch {engine.human_size(pre.download_bytes)} onto disk and {how}?", ui=ui):
        return cfg, None
    if info.os != "macos":
        ui.say("Checking Ollama's MLX engine (installs a ~1 GB bundle if it is missing)…")
    mlx.ensure_mlx(info, sys=sys)
    hf.download(repo, dest, token, sys=sys)
    engine.import_model(
        cfg.engine_url,
        ref.raw,
        dest,
        key=cfg.engine_key,
        quant=None if method is not None else quant,
        sys=sys,
    )
    # Ollama holds its own copy now; a failed import above keeps `dest` for a resumed retry.
    shutil.rmtree(dest, ignore_errors=True)
    with contextlib.suppress(OSError):
        # `<org>/` is left behind by the repo directory above it. `rmdir` refuses a
        # non-empty directory, so another repo from the same org is never touched.
        dest.parent.rmdir()
    return cfg, repo


# Ollama's name charset, with an optional `org/` in front. Each part starts alphanumeric,
# so `.`, `..` and `./x` are refused here rather than reaching `ollama create`. No tag: the
# tag is Ollama's to fill in (`:latest`), and `cfg.model` should hold the plain name asked for.
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?")


def _local_weights(source: Path) -> list[Path]:
    """The `*.safetensors` at the folder's top level, or nothing if it is not a folder."""
    return sorted(source.glob("*.safetensors")) if source.is_dir() else []


def check_local(source: Path, name: str) -> None:
    """Refuse a folder or a name that could never import — the two cheap checks.

    Pure and idempotent, so `model_import` can run it before installing an engine: a
    typo'd path must not be the reason a machine ends up with Ollama on it. It is given
    the path as the user typed it, so the message names what they typed.
    """
    if not _local_weights(source) or not (source / "config.json").is_file():
        raise FriendlyError(
            f"'{source}' does not look like a safetensors model folder "
            "(it needs config.json and *.safetensors).",
            "Point at the folder `hf download` wrote, or run `lepika model add <org>/<repo>` "
            "to fetch one.",
        )
    if not _NAME_RE.fullmatch(name):
        raise FriendlyError(
            f"'{name}' is not a valid model name.",
            "Use letters, digits, . _ - and at most one slash, e.g. --name Qwen/Qwen3.5-2B.",
        )


def import_local(
    info: Machine,
    cfg: config.Config,
    source: Path,
    name: str,
    quant: str,
    *,
    sys: System,
    ui: UI,
) -> str:
    """Import weights the user already has on disk; the name Ollama now serves.

    Nothing here downloads, writes into `source`, or touches `~/.lepika/hf` — the
    folder is the user's, and this command only reads it.
    """
    check_local(source, name)
    weights = _local_weights(source)
    installed = engine.list_models(
        cfg.engine_url, key=cfg.engine_key, managed=cfg.engine_managed, sys=sys
    )
    if any(engine.same_model(served, name) for served, _size in installed):
        # Same reasoning as `import_repo`: rebuilding what Ollama already serves is
        # minutes of quantization for a model that is already there.
        ui.say(f"{escape(name)} is already imported.")
        return name
    # Already-quantized weights import as-is; `-q` would make Ollama refuse them.
    method = hf.quant_method(hf.load_config(source))
    raw_bytes = sum(f.stat().st_size for f in weights)
    quantized = raw_bytes if method is not None else raw_bytes // 4
    # Only Ollama's store is at stake: the weights are already on disk and stay there,
    # so the copy Ollama writes is the one thing that needs room.
    store = mlx.ollama_store(sys=sys)
    free = sys.disk_free(store if store.exists() else paths.lepika_home())
    if free < quantized:
        raise FriendlyError(
            f"Not enough disk: importing {name} writes ~{engine.human_size(quantized)} "
            f"({engine.human_size(free)} available).",
            "Free some space or pick a smaller model.",
        )
    _ram_advisory(name, quantized, info, ui=ui)
    how = (
        f"as-is (already {escape(method)}-quantized, ~{engine.human_size(quantized)})"
        if method is not None
        else f"as {quant} (~{engine.human_size(quantized)})"
    )
    # No confirm: nothing is downloaded, so there is no size to agree to first.
    ui.say(f"Importing {escape(name)} from {escape(str(source))} {how}…")
    if info.os != "macos":
        ui.say("Checking Ollama's MLX engine (installs a ~1 GB bundle if it is missing)…")
    mlx.ensure_mlx(info, sys=sys)
    engine.import_model(
        cfg.engine_url,
        name,
        source,
        key=cfg.engine_key,
        quant=None if method is not None else quant,
        owned=False,
        sys=sys,
    )
    return name


def acquire(
    info: Machine,
    cfg: config.Config,
    ref: ModelRef,
    quant: str = engine.IMPORT_QUANT,
    *,
    sys: System,
    ui: UI,
) -> tuple[config.Config, str | None]:
    """Get `ref` into Ollama — pulled or imported — and return the name it serves.

    Returned with the config as it now is on disk: an import may have saved a typed
    token, which the caller's own later save must keep. None means the user
    declined the download. Ollama decides the format of an
    `hf.co/…` ref (NotGGUF falls through to an import); the file list decides a
    bare `org/repo` (rule 10). `quant` only reaches an import — a pull is a build
    someone else already quantized.
    """
    if ref.kind == "hf_repo":
        return import_repo(info, cfg, ref, quant=quant, sys=sys, ui=ui)
    try:
        engine.pull_model(
            cfg.engine_url,
            ref,
            key=cfg.engine_key,
            managed=cfg.engine_managed,
            sys=sys,
            ui=ui,
        )
    except engine.NotGGUF:
        if ref.kind != "hf_gguf" or not serving.import_allowed(cfg, info):
            raise
        # `hf.co/<org>/<repo>:tag` — the tag is Ollama's, not part of the repo id.
        repo, _tag = gguf.split_tag(ref.raw)
        ui.say(f"{escape(ref.raw)} ships full weights — importing it into Ollama instead.")
        return import_repo(info, cfg, models.parse_model_ref(repo), quant=quant, sys=sys, ui=ui)
    if ref.kind == "hf_gguf" and cfg.engine_managed:
        # Registry models are Ollama-curated; only hf.co pulls carry community
        # templates — sometimes stripped of their tool handling, which turns every
        # request that carries tools into a 400. A managed engine only: `ollama
        # create` cannot send a remote engine's key (rule 9: not ours to fix).
        _ensure_tools(cfg, ref.raw, sys=sys, ui=ui)
    return cfg, ref.raw


def _ensure_tools(cfg: config.Config, name: str, *, sys: System, ui: UI) -> None:
    """Rebuild a pulled ChatML model whose template lost its tool handling.

    Best effort at every step: the pull already succeeded, so nothing here is
    allowed to fail it — an engine that won't answer skips the repair, a template
    in a format LePika does not curate gets one line naming the limitation.
    """
    shown = engine.show(cfg.engine_url, name, key=cfg.engine_key, sys=sys)
    if shown is None:
        return
    capabilities, template = shown
    if not capabilities or "tools" in capabilities:
        return
    if "<|im_start|>" not in template:
        # Not ChatML: the curated template would change the wire format, not fix it.
        ui.say(
            f"[yellow]{escape(name)} has no tool support — chat works, but features "
            "that send tools (web search, code interpreter) will fail with it.[/yellow]"
        )
        return
    try:
        engine.retemplate(
            cfg.engine_url,
            name,
            engine.chatml_tools_template(),
            key=cfg.engine_key,
            sys=sys,
        )
    except FriendlyError as exc:
        ui.say(f"[yellow]{escape(exc.problem)} {escape(exc.fix)}[/yellow]")
        return
    ui.say(
        f"✓ {escape(name)} shipped without tool support — rebuilt its chat template "
        "(ChatML + tools)."
    )
