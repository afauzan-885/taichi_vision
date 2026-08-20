"""Regression tests for owner-local CUDA context binding."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

_engine_spec = importlib.util.spec_from_file_location(
    "taichi_vision.taichi_aot.engine_cuda_owner_test",
    Path(__file__).parents[1] / "engine.py",
)
assert _engine_spec and _engine_spec.loader
engine = importlib.util.module_from_spec(_engine_spec)
sys.modules[_engine_spec.name] = engine
_engine_spec.loader.exec_module(engine)


def test_operation_binds_only_owning_cuda_engine(monkeypatch):
    calls = []

    class Owner:
        arch = "cuda"
        device_id = 7

    monkeypatch.setattr(engine, "_AUTO_DESTROY_ENABLED", True)
    monkeypatch.setattr(engine, "ensure_cuda_context", lambda device_id: calls.append(device_id))
    token = engine._op_begin("owner-local", engine=Owner())
    engine._op_end(token)

    assert calls == [7]


def test_operation_tracking_does_not_scan_global_engine_instances():
    source = (Path(__file__).parents[1] / "engine.py").read_text(encoding="utf-8")
    start = source.index("def _op_begin")
    end = source.index("def _op_end", start)
    block = source[start:end]
    assert "AOTEngine._instances" not in block
    assert 'getattr(engine, "device_id", 0)' in block
