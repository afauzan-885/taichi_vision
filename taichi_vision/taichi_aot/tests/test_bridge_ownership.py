"""Regression tests for process-global native bridge ownership."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load_engine():
    path = Path(__file__).parents[1] / "engine.py"
    name = "taichi_vision.taichi_aot.engine_bridge_owner_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module._WATCHDOG_STOP.set()
    watchdog = getattr(module, "_watchdog", None)
    if watchdog is not None:
        watchdog.join(timeout=2)
    return module


def test_incompatible_bridge_reentry_is_rejected(monkeypatch: pytest.MonkeyPatch):
    engine = _load_engine()
    monkeypatch.setattr(
        engine,
        "detect_target",
        lambda **_kwargs: SimpleNamespace(target_id="vulkan-target"),
    )
    engine._LIB = object()
    engine._BRIDGE_BACKEND = "cpu"
    engine._BRIDGE_TARGET_ID = "cpu-target"

    with pytest.raises(RuntimeError, match="bridge ownership conflict"):
        engine._init_aot_bridge("vulkan")


def test_compatible_bridge_reentry_is_idempotent(monkeypatch: pytest.MonkeyPatch):
    engine = _load_engine()
    monkeypatch.setattr(
        engine,
        "detect_target",
        lambda **_kwargs: SimpleNamespace(target_id="cpu-target"),
    )
    bridge = object()
    engine._LIB = bridge
    engine._BRIDGE_BACKEND = "cpu"
    engine._BRIDGE_TARGET_ID = "cpu-target"
    engine._init_aot_bridge("cpu")
    assert engine._LIB is bridge
