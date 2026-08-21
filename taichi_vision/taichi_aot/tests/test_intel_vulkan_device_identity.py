"""Regression tests for device-scoped Intel Vulkan qualification."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


_CAPABILITIES_PATH = Path(__file__).resolve().parents[1] / "capabilities.py"
_CAPABILITIES_NAME = "taichi_aot_capabilities_test_probe"
_SPEC = importlib.util.spec_from_file_location(_CAPABILITIES_NAME, _CAPABILITIES_PATH)
assert _SPEC is not None and _SPEC.loader is not None
capabilities = importlib.util.module_from_spec(_SPEC)
sys.modules[_CAPABILITIES_NAME] = capabilities
_SPEC.loader.exec_module(capabilities)


def _install_probe(monkeypatch, callback):
    module = types.ModuleType("taichi_vision.vulkan_probe")
    module.intel_vulkan_is_validated = callback
    monkeypatch.setitem(sys.modules, "taichi_vision.vulkan_probe", module)


def _desktop_environment(monkeypatch):
    monkeypatch.delenv("ANDROID_ROOT", raising=False)
    monkeypatch.delenv("ANDROID_DATA", raising=False)
    monkeypatch.delenv("ANDROID_ARGUMENT", raising=False)


def test_intel_policy_uses_evaluated_ordinal_not_aot_device(monkeypatch):
    _desktop_environment(monkeypatch)
    monkeypatch.setenv("AOT_DEVICE", "0")
    calls = []

    def validated(*, device_id):
        calls.append(device_id)
        return device_id == 1

    _install_probe(monkeypatch, validated)
    device = {
        "name": "Intel Arc Test Adapter",
        "vendor": "intel",
        "ordinal": 1,
        "api_version": "1.3",
        "features": {"compute": True, "ssbo": True},
        "capability_source": "vulkan-probe",
    }

    result = capabilities.classify_device(device, "vulkan")
    candidates = capabilities.backend_candidates(device)

    assert result.safe is True
    assert candidates == ["vulkan", "opengl", "cpu"]
    assert calls == [1, 1]


def test_changing_global_aot_device_cannot_change_explicit_device_result(monkeypatch):
    _desktop_environment(monkeypatch)
    calls = []

    def validated(*, device_id):
        calls.append(device_id)
        return device_id == 2

    _install_probe(monkeypatch, validated)
    device = {
        "name": "Intel UHD Test Adapter",
        "vendor": "intel",
        "ordinal": 2,
        "api_version": "1.3",
        "features": {"compute": True, "ssbo": True},
        "capability_source": "vulkan-probe",
    }

    monkeypatch.setenv("AOT_DEVICE", "0")
    first = capabilities.classify_device(device, "vulkan")
    monkeypatch.setenv("AOT_DEVICE", "7")
    second = capabilities.classify_device(device, "vulkan")

    assert first.safe is True
    assert second.safe is True
    assert calls == [2, 2]


def test_intel_device_without_exact_ordinal_fails_closed(monkeypatch):
    _desktop_environment(monkeypatch)
    monkeypatch.setenv("AOT_DEVICE", "5")

    def should_not_be_called(**_kwargs):
        raise AssertionError("qualification must not borrow AOT_DEVICE")

    _install_probe(monkeypatch, should_not_be_called)

    result = capabilities.classify_device(
        {
            "name": "Intel Arc Test Adapter",
            "api_version": "1.3",
            "features": {"compute": True, "ssbo": True},
            "capability_source": "vulkan-probe",
        },
        "vulkan",
    )
    candidates = capabilities.backend_candidates("Intel Arc Test Adapter")

    assert result.safe is False
    assert "exact evaluated device identity" in result.reason
    assert candidates == ["opengl", "cpu"]


def test_invalid_evaluated_ordinal_fails_closed(monkeypatch):
    _desktop_environment(monkeypatch)

    def should_not_be_called(**_kwargs):
        raise AssertionError("invalid ordinal must not reach qualification")

    _install_probe(monkeypatch, should_not_be_called)
    device = {
        "name": "Intel Test Adapter",
        "vendor": "intel",
        "ordinal": "not-an-integer",
    }

    assert capabilities.classify_device(device, "vulkan").safe is False
    assert capabilities.backend_candidates(device) == ["opengl", "cpu"]
