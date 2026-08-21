"""End-to-end contracts for canonical graphics capability admission."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from taichi_vision.taichi_aot import capabilities
from taichi_vision.taichi_aot.gfx_capabilities import (
    negotiate_graphics_capabilities,
    unknown_graphics_snapshot,
)


def _load_engine_source():
    path = Path(__file__).parents[1] / "engine.py"
    name = "taichi_vision.taichi_aot.engine_graphics_authority_test"
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


def test_backend_classifier_reuses_supplied_snapshot(monkeypatch: pytest.MonkeyPatch):
    snapshot = negotiate_graphics_capabilities(
        "vulkan",
        {
            "api_version": "1.3",
            "features": {"compute": True, "ssbo": True},
            "capability_source": "vulkan-probe",
        },
    )

    def fail_if_renegotiated(*args, **kwargs):
        raise AssertionError("capability snapshot was negotiated twice")

    monkeypatch.setattr(capabilities, "negotiate_graphics_capabilities", fail_if_renegotiated)
    result = capabilities.classify_device(
        {
            "name": "NVIDIA RTX canonical",
            "vendor": "NVIDIA",
            "capability_snapshot": snapshot,
        },
        "vulkan",
    )
    assert result.safe is True


def test_snapshot_cache_identity_changes_after_driver_change(monkeypatch: pytest.MonkeyPatch):
    engine = _load_engine_source()
    evidence = {
        "api_version": "1.3",
        "features": {"compute": True, "ssbo": True},
        "capability_source": "vulkan-probe",
        "fingerprint": "device-d1",
        "driver_uuid": "driver-d1",
        "driver_version": "1",
    }
    monkeypatch.setattr(engine, "query_vulkan_capability_snapshot", lambda _id: dict(evidence))
    first = engine._graphics_capability_snapshot("vulkan", 0, "GPU")
    second = engine._graphics_capability_snapshot("vulkan", 0, "GPU")
    assert first is second

    evidence["fingerprint"] = "device-d2"
    evidence["driver_uuid"] = "driver-d2"
    third = engine._graphics_capability_snapshot("vulkan", 0, "GPU")
    assert third is not first


def test_explicit_graphics_backend_rejects_missing_probe(monkeypatch: pytest.MonkeyPatch):
    engine = _load_engine_source()
    monkeypatch.setattr(engine, "get_vulkan_device_name", lambda _id: "NVIDIA RTX")
    monkeypatch.setattr(engine, "scan_vulkan_device_records", lambda: [])
    monkeypatch.setattr(
        engine,
        "_graphics_capability_snapshot",
        lambda *_args: unknown_graphics_snapshot("vulkan", "probe unavailable"),
    )
    monkeypatch.setenv("AOT_AUTOSCAN", "0")
    with pytest.raises(RuntimeError, match="lacks qualified capability evidence"):
        engine.resolve_backend_config(arch="vulkan", device_id=0)


def test_explicit_graphics_backend_carries_same_snapshot_into_config(monkeypatch: pytest.MonkeyPatch):
    engine = _load_engine_source()
    snapshot = negotiate_graphics_capabilities(
        "vulkan",
        {
            "api_version": "1.3",
            "features": {"compute": True, "ssbo": True},
            "capability_source": "vulkan-probe",
        },
    )
    monkeypatch.setattr(engine, "get_vulkan_device_name", lambda _id: "NVIDIA RTX")
    monkeypatch.setattr(engine, "scan_vulkan_device_records", lambda: [])
    monkeypatch.setattr(engine, "_graphics_capability_snapshot", lambda *_args: snapshot)
    monkeypatch.setenv("AOT_AUTOSCAN", "0")
    config = engine.resolve_backend_config(arch="vulkan", device_id=0)
    assert config.capability_snapshot is snapshot
