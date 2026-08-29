"""The suite's own safety rails (docs/architecture.md rule 3)."""

from __future__ import annotations

import re
import urllib.request

import pytest
from conftest import RealNetworkCall

from lepika import express


def test_real_urlopen_is_replaced_for_every_test() -> None:
    with pytest.raises(RealNetworkCall, match=re.escape("http://example.invalid/")):
        urllib.request.urlopen("http://example.invalid/", timeout=0.1)


def test_the_guard_also_catches_a_keyword_url() -> None:
    with pytest.raises(RealNetworkCall, match=re.escape("http://example.invalid/")):
        urllib.request.urlopen(url="http://example.invalid/", timeout=0.1)


def test_the_guard_is_not_swallowed_by_except_exception() -> None:
    # webui_up catches Exception and returns False; the guard must still fail the test.
    with pytest.raises(RealNetworkCall):
        express.webui_up(3000)
