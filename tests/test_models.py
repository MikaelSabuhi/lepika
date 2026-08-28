from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from lepika import models
from lepika.errors import FriendlyError


def test_parse_plain_tag_is_ollama() -> None:
    ref = models.parse_model_ref("qwen3:8b")
    assert ref == models.ModelRef(raw="qwen3:8b", kind="ollama")


def test_parse_hf_gguf_url_normalizes_scheme_and_domain() -> None:
    ref = models.parse_model_ref("https://huggingface.co/unsloth/gemma-3-4b-it-GGUF")
    assert ref.kind == "hf_gguf"
    assert ref.raw == "hf.co/unsloth/gemma-3-4b-it-GGUF"


def test_parse_bare_hf_co_prefix() -> None:
    ref = models.parse_model_ref("hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M")
    assert ref.kind == "hf_gguf"


def test_parse_org_slash_repo_is_hf_repo() -> None:
    ref = models.parse_model_ref("meta-llama/Llama-3.3-70B-Instruct")
    assert ref.kind == "hf_repo"


def test_parse_empty_raises() -> None:
    with pytest.raises(FriendlyError):
        models.parse_model_ref("   ")


def test_load_curated_falls_back_to_bundled_on_network_error(isolated_home: Any) -> None:
    def boom(url: str, timeout: float = 0) -> Any:
        raise OSError("no network")

    curated = models.load_curated(urlopen=boom)
    assert len(curated) >= 5
    assert all(m.min_ram_gb > 0 for m in curated)


def test_load_curated_skips_network_when_disabled() -> None:
    def fail_if_called(url: str, timeout: float = 0) -> Any:
        raise AssertionError("network should not be used")

    curated = models.load_curated(fetch_remote=False, urlopen=fail_if_called)
    assert curated


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> bytes:
        return self._text.encode("utf-8")


def _serving(text: str) -> Callable[..., Any]:
    """A urlopen stand-in that serves `text` as the remote models.toml."""

    def opener(url: str, timeout: float = 0) -> Any:
        return _FakeResponse(text)

    return opener


def test_load_curated_falls_back_to_bundled_on_invalid_remote_toml() -> None:
    curated = models.load_curated(urlopen=_serving("this is [not valid toml"))
    assert len(curated) >= 5
    assert any(m.ref == "qwen3:8b" for m in curated)


def test_load_curated_falls_back_to_bundled_when_remote_has_no_models() -> None:
    curated = models.load_curated(urlopen=_serving("schema = 2\n"))
    assert len(curated) >= 5
    assert any(m.ref == "qwen3:8b" for m in curated)


def test_load_curated_ignores_unknown_keys_in_remote_entries() -> None:
    remote = (
        "schema = 2\n"
        "\n"
        "[[models]]\n"
        'name = "Future model"\n'
        'ref = "future:1b"\n'
        "min_ram_gb = 4\n"
        'notes = "from the future"\n'
        "context_window = 128000\n"
        'tags = ["experimental"]\n'
    )
    curated = models.load_curated(urlopen=_serving(remote))
    assert curated == [
        models.CuratedModel(
            name="Future model", ref="future:1b", min_ram_gb=4, notes="from the future"
        )
    ]


def test_load_curated_skips_remote_entries_missing_required_fields() -> None:
    remote = (
        "[[models]]\n"
        'name = "Good"\n'
        'ref = "good:1b"\n'
        "min_ram_gb = 4\n"
        "\n"
        "[[models]]\n"
        'name = "Broken — no ref"\n'
        "min_ram_gb = 8\n"
    )
    curated = models.load_curated(urlopen=_serving(remote))
    assert [m.ref for m in curated] == ["good:1b"]


def test_load_curated_skips_remote_entries_with_wrong_field_types() -> None:
    """A plausible TOML typo (quoted number) must not reach `fitting` and crash there."""
    remote = (
        "[[models]]\n"
        'name = "Good"\n'
        'ref = "good:1b"\n'
        "min_ram_gb = 4\n"
        "\n"
        "[[models]]\n"
        'name = "Quoted number"\n'
        'ref = "typo:8b"\n'
        'min_ram_gb = "8"\n'
        "\n"
        "[[models]]\n"
        "name = 42\n"
        'ref = "notaname:1b"\n'
        "min_ram_gb = 4\n"
        "\n"
        "[[models]]\n"
        'name = "Boolean ram"\n'
        'ref = "bool:1b"\n'
        "min_ram_gb = true\n"
    )
    curated = models.load_curated(urlopen=_serving(remote))
    assert [m.ref for m in curated] == ["good:1b"]
    assert models.fitting(curated, ram_gb=16.0) == curated


def test_fitting_filters_by_ram() -> None:
    small = models.CuratedModel(name="S", ref="s:1b", min_ram_gb=2)
    big = models.CuratedModel(name="B", ref="b:70b", min_ram_gb=48)
    assert models.fitting([small, big], ram_gb=16.0) == [small]
