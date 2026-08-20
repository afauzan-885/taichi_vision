"""Regression test for serialized AOTEngine construction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
import time


_spec = importlib.util.spec_from_file_location(
    "taichi_vision.taichi_aot.engine_singleton_test",
    Path(__file__).parents[1] / "engine.py",
)
assert _spec and _spec.loader
engine = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = engine
_spec.loader.exec_module(engine)


def test_constructor_lock_serializes_same_key_acquisition(monkeypatch):
    active = 0
    maximum = 0
    calls = 0
    state_lock = threading.Lock()
    result = object()
    results = []

    def fake_unlocked(cls, arch=None, device_id=None):
        nonlocal active, maximum, calls
        with state_lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.03)
            key = (arch, device_id)
            if key not in cls._instances:
                cls._instances[key] = result
                calls += 1
            results.append(cls._instances[key])
            return cls._instances[key]
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(engine.AOTEngine, "_new_unlocked", classmethod(fake_unlocked))
    engine.AOTEngine._instances.clear()
    threads = [
        threading.Thread(
            target=lambda: engine.AOTEngine.__new__(engine.AOTEngine, "cpu", 0)
        )
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert calls == 1
    assert maximum == 1
    assert len(results) == 8
    assert all(item is result for item in results)
