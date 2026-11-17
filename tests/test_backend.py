"""The `Backend` contract: both real backends, the fake, and the one place that picks."""

from __future__ import annotations

import inspect
from typing import Any

import fakes
import pytest
from fakes import Backends, FakeBackend, FakeSystem, FakeUI, UnprovidedBackend

from lepika import backend, config, detect, express, server
from lepika.backend import Backend
from lepika.express import ExpressBackend
from lepika.server import ServerBackend

_METHODS = (
    "start",
    "stop",
    "update",
    "logs",
    "notes",
    "apply_engine_change",
    "ensure_engine",
    "release",
)


def _shape(func: Any) -> list[tuple[str, Any, Any]]:
    # Names, kinds and defaults: annotations print differently between interpreters.
    return [(p.name, p.kind, p.default) for p in inspect.signature(func).parameters.values()]


def test_both_backends_and_the_fake_match_the_protocol_signatures() -> None:
    declared = {
        name
        for name, value in vars(Backend).items()
        if not name.startswith("_") and callable(value)
    }
    assert declared == set(_METHODS)
    for name in _METHODS:
        expected = _shape(getattr(Backend, name))
        for cls in (ExpressBackend, ServerBackend, FakeBackend):
            assert _shape(getattr(cls, name)) == expected, (cls.__name__, name)
    for cls in (ExpressBackend, ServerBackend):
        assert _shape(cls.__init__) == [
            ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
            ("sys", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
            ("ui", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
        ]


def test_backend_for_picks_the_class_for_the_configs_mode_and_hands_it_both_seams() -> None:
    machine, face = FakeSystem(), FakeUI()
    chosen = backend.backend_for(config.Config(mode="server"), sys=machine, ui=face)
    assert isinstance(chosen, ServerBackend)
    assert chosen.sys is machine and chosen.ui is face
    chosen = backend.backend_for(config.Config(mode="express"), sys=machine, ui=face)
    assert isinstance(chosen, ExpressBackend)
    assert chosen.sys is machine and chosen.ui is face
    # Selecting runs nothing: the backends only act when a method is called.
    assert machine.ran == [] and machine.looked_up == [] and machine.requested == []


@pytest.mark.parametrize("module", [express, server])
@pytest.mark.parametrize("name", ["start_stack", "stop", "update", "logs"])
def test_the_old_module_level_lifecycle_functions_are_gone(
    module: Any, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A test still patching one would otherwise pass while the real backend ran."""
    with pytest.raises(AttributeError):
        monkeypatch.setattr(module, name, lambda *a, **k: None)


def test_the_fake_records_every_call_and_returns_what_it_was_given() -> None:
    hooked: list[str] = []
    fake = FakeBackend(url="u", was_running=False, steps=["s"], sections=[("t", "x")])
    info = object()
    cfg = config.Config()
    assert fake.start(info, cfg, after_engine=lambda: hooked.append("hook")) == "u"
    assert fake.stop(info, cfg) is False
    assert fake.update(info, cfg) == ["s"]
    assert fake.logs(7) == [("t", "x")]
    # Unscripted, the rest are a quiet backend: no notes, a restart, nothing freed.
    assert fake.notes(info, cfg) == []
    assert fake.apply_engine_change(info, cfg) is True
    assert fake.ensure_engine(info, cfg) is None
    assert fake.release(info, cfg) is None
    # The hook runs, as both real starts run it once the engine is up.
    assert hooked == ["hook"]
    assert [(i, c) for i, c, _ in fake.started] == [(info, cfg)]
    assert fake.stopped == [(info, cfg)] and fake.updated == [(info, cfg)]
    assert fake.logged == [7]
    assert fake.noted == [(info, cfg)] and fake.engine_changes == [(info, cfg)]
    assert fake.engines_ensured == [(info, cfg)] and fake.released == [(info, cfg)]


def test_the_fake_hands_each_call_to_the_tests_own_behaviour() -> None:
    seen: list[Any] = []
    fake = FakeBackend(
        on_start=lambda info, cfg, **k: seen.append(("start", k)) or "custom",
        on_stop=lambda info, cfg: seen.append("stop") or True,
        on_update=lambda info, cfg: seen.append("update") or [],
        on_logs=lambda lines: seen.append(("logs", lines)) or [],
        on_notes=lambda info, cfg: seen.append("notes") or ["a note"],
        on_apply_engine_change=lambda info, cfg: seen.append("change") or False,
        on_ensure_engine=lambda info, cfg: seen.append("ensure"),
        on_release=lambda info, cfg: seen.append("release") or "freed",
    )
    cfg = config.Config()
    assert fake.start(None, cfg) == "custom"
    assert fake.stop(None, cfg) is True
    assert fake.update(None, cfg) == []
    assert fake.logs(3) == []
    assert fake.notes(None, cfg) == ["a note"]
    assert fake.apply_engine_change(None, cfg) is False
    assert fake.ensure_engine(None, cfg) is None
    assert fake.release(None, cfg) == "freed"
    assert seen == [
        ("start", {"after_engine": None}),
        "stop",
        "update",
        ("logs", 3),
        "notes",
        "change",
        "ensure",
        "release",
    ]
    assert len(fake.started) == 1 and len(fake.stopped) == 1
    assert len(fake.noted) == 1 and len(fake.engine_changes) == 1
    assert len(fake.engines_ensured) == 1 and len(fake.released) == 1


def test_a_mode_the_test_gave_no_backend_for_fails_loudly() -> None:
    """A BaseException: no `except Exception` in `src/` may read it as a quiet failure."""
    pick = Backends(server=FakeBackend())
    assert (
        pick(config.Config(mode="server"), sys=FakeSystem(), ui=FakeUI()) is pick.by_mode["server"]
    )
    with pytest.raises(UnprovidedBackend):
        pick(config.Config(mode="express"), sys=FakeSystem(), ui=FakeUI())
    assert not issubclass(UnprovidedBackend, Exception)
    assert pick.selected == ["server", "express"]


def test_deps_defaults_to_fresh_fakes_and_the_real_backends_over_them() -> None:
    """What an un-patched test used to reach: the real backends, here over fakes."""
    deps = fakes.deps()
    assert isinstance(deps.sys, FakeSystem) and isinstance(deps.ui, FakeUI)
    assert fakes.deps().sys is not deps.sys and fakes.deps().ui is not deps.ui
    for mode, kind in (("express", ExpressBackend), ("server", ServerBackend)):
        chosen = deps.backends(config.Config(mode=mode))
        assert isinstance(chosen, kind)
        assert chosen.sys is deps.sys and chosen.ui is deps.ui


def test_deps_refuses_to_detect_a_machine_the_test_did_not_choose() -> None:
    """A BaseException, like the other refusals: no `except Exception` may hide it."""
    with pytest.raises(fakes.UnprovidedMachine):
        fakes.deps().machine()
    assert not issubclass(fakes.UnprovidedMachine, Exception)
    mac = detect.Machine("macos", "arm64", "apple", 16.0)
    assert fakes.deps(machine=mac).machine() is mac


def test_deps_hands_through_what_the_test_gave() -> None:
    machine, face, chosen = FakeSystem(), FakeUI(), FakeBackend()
    pick = Backends(express=chosen)
    deps = fakes.deps(sys=machine, ui=face, backends=pick)
    assert deps.sys is machine and deps.ui is face
    assert deps.backends(config.Config()) is chosen
    assert pick.selected == ["express"]
    with pytest.raises(UnprovidedBackend):
        deps.backends(config.Config(mode="server"))
