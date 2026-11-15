"""LePika's reading of real upstream output, against the real upstreams.

The unit suite fakes Ollama, the `hf` CLI and `ps`, so it only proves the code agrees
with its own assumptions. These tests call the same functions production does and
assert on LePika's classification, so a wording change upstream fails here instead of
in a user's terminal. They need a running Ollama on 127.0.0.1:11434 and the network,
and are deselected from the default run.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from lepika import engine, express, hf, models, proc, system
from lepika.errors import FriendlyError

pytestmark = pytest.mark.canary

OLLAMA_URL = "http://127.0.0.1:11434"
# Safetensors only, a few megabytes, and maintained by Hugging Face for exactly this.
SAFETENSORS_REPO = "hf-internal-testing/tiny-random-gpt2"
GATED_REPO = "meta-llama/Llama-3.2-1B"


def _no_stored_token(tmp_path: Path) -> dict[str, str]:
    """The environment of a user who never logged in: no HF_TOKEN, and an `hf` home
    without the token file a developer's own `hf auth login` would have left."""
    env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
    env["HF_HOME"] = str(tmp_path / "hf-home")
    return env


def test_pull_of_the_smallest_curated_model_succeeds() -> None:
    curated = models.load_curated(fetch_remote=False, sys=system.REAL)
    smallest = min(curated, key=lambda m: m.min_ram_gb)
    engine.pull_model(OLLAMA_URL, models.parse_model_ref(smallest.ref))
    installed = [name for name, _size in engine.list_models(OLLAMA_URL)]
    assert any(engine.same_model(name, smallest.ref) for name in installed), installed


def test_a_repo_without_gguf_is_classified_not_gguf() -> None:
    ref = models.parse_model_ref(f"hf.co/{SAFETENSORS_REPO}")
    with pytest.raises(engine.NotGGUF):
        engine.pull_model(OLLAMA_URL, ref)


def test_preflight_of_a_public_repo_lists_files(tmp_path: Path) -> None:
    found = hf.preflight(SAFETENSORS_REPO, environ=_no_stored_token(tmp_path))
    assert isinstance(found, hf.Preflight)
    assert len(found.files) >= 1


def test_preflight_of_a_missing_repo_is_not_found(tmp_path: Path) -> None:
    with pytest.raises(FriendlyError) as raised:
        hf.preflight(
            "lepika-canary-no-such-org/no-such-repo-0000", environ=_no_stored_token(tmp_path)
        )
    # Exactly the not-found branch: a GatedRepo or the generic listing failure would
    # send the user to accept a licence or check their connection instead.
    assert type(raised.value) is FriendlyError
    assert "is not on Hugging Face" in raised.value.problem


def test_preflight_of_a_gated_repo_is_gated(tmp_path: Path) -> None:
    with pytest.raises(hf.GatedRepo):
        hf.preflight(GATED_REPO, token="", environ=_no_stored_token(tmp_path))


def test_a_running_ollama_is_recognised_by_pid() -> None:
    found = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True, text=True, check=False)
    pids = [int(pid) for pid in found.stdout.split()]
    assert pids, "no process named ollama is running"
    assert express._is_ollama_process(pids[0], "linux", proc.run_logged) is True
    assert express._is_ollama_process(os.getpid(), "linux", proc.run_logged) is False


def test_the_engine_reports_a_version_lepika_can_compare() -> None:
    parsed = engine._version_tuple(engine.version(OLLAMA_URL))
    assert len(parsed) >= 3
    assert all(isinstance(part, int) for part in parsed)
    # `_version_tuple` pads with zeros, so a version string with no numbers in it
    # would still give three: all zeros is how that failure shows.
    assert parsed > (0, 0, 0)
