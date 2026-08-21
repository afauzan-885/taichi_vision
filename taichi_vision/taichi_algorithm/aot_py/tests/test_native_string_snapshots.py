"""Regression guards for stable native diagnostic/device string snapshots."""

from __future__ import annotations

from pathlib import Path


SOURCE = (Path(__file__).parents[1] / "taichi_aot_engine.cpp").read_text(encoding="utf-8")


def test_exported_mutable_strings_use_thread_local_snapshots():
    for function in (
        "get_last_init_error",
        "scan_vulkan_devices",
        "get_runtime_device_name",
        "get_runtime_context_backend",
        "get_last_engine_error",
    ):
        start = SOURCE.index(f"EXPORT const char *{function}")
        end = SOURCE.find("EXPORT ", start + 8)
        block = SOURCE[start : end if end >= 0 else len(SOURCE)]
        assert "static thread_local std::string" in block, function
