"""Regression tests for fail-closed stable device selection."""

from __future__ import annotations

from taichi_vision.device_selection import device_fingerprint, resolve_device_selector


def _identical_devices():
    return [
        {
            "ordinal": 0,
            "name": "NVIDIA RTX Test",
            "vendor_id": 4318,
            "device_id": 1234,
            "driver_id": "native",
            "native": True,
        },
        {
            "ordinal": 1,
            "name": "NVIDIA RTX Test",
            "vendor_id": 4318,
            "device_id": 1234,
            "driver_id": "native",
            "native": True,
        },
    ]


def test_identical_fingerprint_does_not_use_enumeration_order():
    devices = _identical_devices()
    selector = {"fingerprint": device_fingerprint(devices[0])}
    assert resolve_device_selector(selector, devices, cached_id=1) is None


def test_identical_name_does_not_use_cached_ordinal():
    devices = _identical_devices()
    selector = {"vendor": "nvidia", "name": "nvidia rtx test", "native": True}
    assert resolve_device_selector(selector, devices, cached_id=1) is None


def test_invalid_serialized_native_flag_fails_closed():
    devices = _identical_devices()[:1]
    selector = {"vendor": "nvidia", "name": "nvidia rtx test", "native": "false"}
    assert resolve_device_selector(selector, devices) is None
