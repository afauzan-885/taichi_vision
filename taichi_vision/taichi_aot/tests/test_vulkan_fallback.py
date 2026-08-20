"""Contract tests for the platform-correct Vulkan identity fallback."""

from __future__ import annotations

from pathlib import Path
import ctypes
import importlib.util
import sys

import pytest


_engine_spec = importlib.util.spec_from_file_location(
    "taichi_vision.taichi_aot.engine_vulkan_fallback_test",
    Path(__file__).parents[1] / "engine.py",
)
assert _engine_spec and _engine_spec.loader
engine = importlib.util.module_from_spec(_engine_spec)
sys.modules[_engine_spec.name] = engine
_engine_spec.loader.exec_module(engine)


class _FakeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class _FakeVulkan:
    def __init__(self, *, enumerate_result=0):
        self.enumerate_result = enumerate_result
        self.destroyed = False
        self.vkCreateInstance = _FakeFunction(self.create_instance)
        self.vkEnumeratePhysicalDevices = _FakeFunction(self.enumerate_devices)
        self.vkGetPhysicalDeviceProperties = _FakeFunction(self.get_properties)
        self.vkDestroyInstance = _FakeFunction(self.destroy_instance)

    def create_instance(self, _create_info, _allocator, instance):
        instance[0] = ctypes.c_void_p(123)
        return 0

    def enumerate_devices(self, _instance, count, devices):
        if self.enumerate_result:
            return self.enumerate_result
        if not devices:
            count[0] = 2
            return 0
        devices[0] = ctypes.c_void_p(10)
        devices[1] = ctypes.c_void_p(11)
        return 0

    def get_properties(self, device, buffer):
        names = {10: b"Adapter A", 11: b"Adapter B"}
        payload = names[int(getattr(device, "value", device))] + b"\0"
        for offset, value in enumerate(payload, start=20):
            buffer[offset] = value

    def destroy_instance(self, _instance, _allocator):
        self.destroyed = True


def test_fallback_uses_platform_loader_and_preserves_ordinal(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeVulkan()
    monkeypatch.setattr(engine, "scan_vulkan_device_records", lambda: [])
    monkeypatch.setattr(engine.ctypes, "CDLL", lambda name: fake)

    assert engine.get_vulkan_device_name(1) == "Adapter B", engine.get_vulkan_device_probe_diagnostic()
    assert engine.get_vulkan_device_probe_diagnostic() == ""


def test_fallback_rejects_vulkan_result_errors_with_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeVulkan(enumerate_result=1)
    monkeypatch.setattr(engine, "scan_vulkan_device_records", lambda: [])
    monkeypatch.setattr(engine.ctypes, "CDLL", lambda name: fake)

    assert engine.get_vulkan_device_name(0) is None
    assert "VkResult 1" in engine.get_vulkan_device_probe_diagnostic()


def test_source_keeps_valid_vulkan_structure_types() -> None:
    source = (Path(__file__).parents[1] / "engine.py").read_text(encoding="utf-8")
    start = source.index("def get_vulkan_device_name")
    fallback = source[start : source.index("# -------------------------------------------------------------------------", start)]
    assert "sType=0" in fallback
    assert "sType=1" in fallback
    assert "vkEnumeratePhysicalDevices(count)" in fallback
