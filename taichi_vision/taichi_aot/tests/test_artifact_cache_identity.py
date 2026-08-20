"""Regression tests for artifact quarantine identity boundaries."""

from __future__ import annotations

import os
from pathlib import Path

from taichi_vision.taichi_aot.artifact_cache import artifact_key


def test_artifact_key_changes_when_content_changes_with_same_metadata(tmp_path: Path) -> None:
    artifact = tmp_path / "graph.tcm"
    artifact.write_bytes(b"first payload")
    stat = artifact.stat()
    first = artifact_key(artifact, "vulkan", 0, "GPU")

    artifact.write_bytes(b"second payload")
    os.utime(artifact, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = artifact_key(artifact, "vulkan", 0, "GPU")

    assert first != second


def test_artifact_key_scopes_driver_and_device_identity(tmp_path: Path) -> None:
    artifact = tmp_path / "graph.tcm"
    artifact.write_bytes(b"stable payload")
    common = {
        "target_id": "vulkan_x86_64_windows_nvidia",
        "device_fingerprint": "nvidia:10de:2684:gpu-a:native",
        "driver_version": "555.1",
        "driver_uuid": "driver-a",
    }

    baseline = artifact_key(artifact, "vulkan", 0, "GPU", **common)
    driver_changed = artifact_key(
        artifact, "vulkan", 0, "GPU", **{**common, "driver_version": "556.1"}
    )
    device_changed = artifact_key(
        artifact,
        "vulkan",
        0,
        "GPU",
        **{**common, "device_fingerprint": "nvidia:10de:2684:gpu-b:native"},
    )

    assert baseline != driver_changed
    assert baseline != device_changed
