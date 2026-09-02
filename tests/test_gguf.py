from __future__ import annotations

import json
import urllib.request
from typing import Any

import pytest

from lepika import gguf

# A trimmed copy of the real listing of unsloth/Qwen3.8-27B-GGUF (2026-09-02):
# a sharded BF16 in its own folder, an MTP draft that would collide with Q4_0,
# projectors and an imatrix that are not chat models, and UD- prefixed quants.
UNSLOTH = {
    "siblings": [
        {"rfilename": ".gitattributes", "size": 1500},
        {"rfilename": "README.md", "size": 20000},
        {"rfilename": "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", "size": 49_986_159_616},
        {"rfilename": "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf", "size": 4_671_576_000},
        {"rfilename": "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "size": 1_369_590_656},
        {"rfilename": "Qwen3.8-27B-Q4_0.gguf", "size": 16_056_478_688},
        {"rfilename": "Qwen3.8-27B-Q8_0.gguf", "size": 29_047_086_048},
        {"rfilename": "Qwen3.8-27B-UD-IQ4_XS.gguf", "size": 14_252_845_984},
        {"rfilename": "Qwen3.8-27B-UD-Q4_K_M.gguf", "size": 16_464_440_224},
        {"rfilename": "Qwen3.8-27B-UD-Q6_K.gguf", "size": 21_983_677_344},
        {"rfilename": "imatrix_unsloth.gguf", "size": 13_642_656},
        {"rfilename": "mmproj-BF16.gguf", "size": 931_146_432},
        {"rfilename": "mmproj-F16.gguf", "size": 927_607_488},
    ]
}


def quants(builds: list[gguf.Build]) -> list[str]:
    return [b.quant for b in builds]


def test_parse_builds_one_row_per_quant_sorted_by_size() -> None:
    builds = gguf.parse_builds(UNSLOTH)
    assert quants(builds) == ["UD-IQ4_XS", "Q4_0", "UD-Q4_K_M", "UD-Q6_K", "Q8_0", "BF16"]


def test_parse_builds_sums_shards_kept_in_a_folder_named_after_the_quant() -> None:
    bf16 = next(b for b in gguf.parse_builds(UNSLOTH) if b.quant == "BF16")
    assert bf16.size_bytes == 49_986_159_616 + 4_671_576_000


def test_parse_builds_ignores_drafts_projectors_and_imatrix() -> None:
    q4_0 = next(b for b in gguf.parse_builds(UNSLOTH) if b.quant == "Q4_0")
    # The MTP/ draft is also "Q4_0" by name; it must neither add to nor replace the real one.
    assert q4_0.size_bytes == 16_056_478_688
    assert "F16" not in quants(gguf.parse_builds(UNSLOTH))  # mmproj-F16 is a projector


def test_parse_builds_root_file_wins_over_a_folder_twin() -> None:
    payload = {
        "siblings": [
            {"rfilename": "Q8_0/x-Q8_0-00001-of-00002.gguf", "size": 10},
            {"rfilename": "Q8_0/x-Q8_0-00002-of-00002.gguf", "size": 10},
            {"rfilename": "x-Q8_0.gguf", "size": 30},
        ]
    }
    assert [(b.quant, b.size_bytes) for b in gguf.parse_builds(payload)] == [("Q8_0", 30)]


def test_parse_builds_skips_folders_that_are_not_the_quant() -> None:
    payload = {"siblings": [{"rfilename": "old/x-Q4_K_M.gguf", "size": 5}]}
    assert gguf.parse_builds(payload) == []


@pytest.mark.parametrize(
    ("name", "quant"),
    [
        ("llama-2-7b.Q4_K_M.gguf", "Q4_K_M"),  # dot-separated (TheBloke style)
        ("gemma-3-4b-it-IQ4_NL.gguf", "IQ4_NL"),
        ("model-UD-Q2_K_XL.gguf", "UD-Q2_K_XL"),
        ("model-q5_k_m.gguf", "q5_k_m"),  # case preserved: it is the tag Ollama pulls
        ("model-MXFP4.gguf", "MXFP4"),
        ("model-F16.gguf", "F16"),
    ],
)
def test_parse_builds_reads_the_quant_from_the_stem(name: str, quant: str) -> None:
    assert quants(gguf.parse_builds({"siblings": [{"rfilename": name, "size": 1}]})) == [quant]


def test_parse_builds_skips_files_with_no_recognisable_quant() -> None:
    payload = {
        "siblings": [
            {"rfilename": "model.gguf", "size": 1},
            {"rfilename": "x-Q4_K_M.bin", "size": 1},
        ]
    }
    assert gguf.parse_builds(payload) == []


def test_parse_builds_skips_malformed_entries() -> None:
    payload = {
        "siblings": [
            "not a dict",
            {"rfilename": "x-Q4_0.gguf"},  # no size
            {"rfilename": "x-Q4_1.gguf", "size": "big"},
            {"rfilename": "x-Q5_0.gguf", "size": True},
            {"rfilename": "x-Q8_0.gguf", "size": 7},
        ]
    }
    assert [(b.quant, b.size_bytes) for b in gguf.parse_builds(payload)] == [("Q8_0", 7)]


@pytest.mark.parametrize("payload", [None, [], "text", {}, {"siblings": "x"}])
def test_parse_builds_refuses_what_is_not_a_listing(payload: Any) -> None:
    with pytest.raises(gguf.Unavailable):
        gguf.parse_builds(payload)


def test_build_size_and_quantized() -> None:
    assert gguf.Build("Q4_K_M", 14_252_845_984).size_gb == pytest.approx(14.25, abs=0.01)
    assert gguf.Build("Q4_K_M", 1).quantized is True
    assert gguf.Build("bf16", 1).quantized is False
    assert gguf.Build("F32", 1).quantized is False


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body


def _serving(payload: Any) -> Any:
    seen: list[urllib.request.Request] = []

    def opener(request: urllib.request.Request, timeout: float = 0) -> _Response:
        seen.append(request)
        return _Response(json.dumps(payload).encode())

    opener.seen = seen  # type: ignore[attr-defined]
    return opener


def test_list_builds_asks_the_hub_for_blob_sizes_without_a_token() -> None:
    opener = _serving(UNSLOTH)
    builds = gguf.list_builds("unsloth/Qwen3.8-27B-GGUF", urlopen=opener)
    assert quants(builds)[0] == "UD-IQ4_XS"
    request = opener.seen[0]
    assert (
        request.full_url == "https://huggingface.co/api/models/unsloth/Qwen3.8-27B-GGUF?blobs=true"
    )
    assert request.get_header("Authorization") is None


def test_list_builds_sends_the_token_as_a_bearer_header() -> None:
    opener = _serving(UNSLOTH)
    gguf.list_builds("unsloth/x-GGUF", token="hf_secret", urlopen=opener)
    assert opener.seen[0].get_header("Authorization") == "Bearer hf_secret"


def test_list_builds_is_unavailable_when_the_hub_does_not_answer() -> None:
    def offline(request: Any, timeout: float = 0) -> Any:
        raise OSError("no network")

    with pytest.raises(gguf.Unavailable):
        gguf.list_builds("unsloth/x-GGUF", urlopen=offline)


def test_list_builds_is_unavailable_on_a_body_that_is_not_json() -> None:
    def html(request: Any, timeout: float = 0) -> _Response:
        return _Response(b"<html>rate limited</html>")

    with pytest.raises(gguf.Unavailable):
        gguf.list_builds("unsloth/x-GGUF", urlopen=html)


def test_list_builds_returns_nothing_for_a_repo_without_gguf_files() -> None:
    payload = {"siblings": [{"rfilename": "model.safetensors", "size": 5}]}
    assert gguf.list_builds("Qwen/Qwen3.5-2B", urlopen=_serving(payload)) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("hf.co/unsloth/Qwen3.8-27B-GGUF", ("unsloth/Qwen3.8-27B-GGUF", "")),
        ("hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M", ("unsloth/Qwen3.8-27B-GGUF", "UD-Q4_K_M")),
        ("HF.CO/unsloth/x-GGUF:Q8_0", ("unsloth/x-GGUF", "Q8_0")),
        ("hf.co/unsloth/x-GGUF:", ("unsloth/x-GGUF", "")),
    ],
)
def test_split_tag(raw: str, expected: tuple[str, str]) -> None:
    assert gguf.split_tag(raw) == expected
