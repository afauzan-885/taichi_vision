"""Regression tests for negotiated graphics evidence and fail-closed admission."""

from __future__ import annotations

from pathlib import Path

from taichi_vision.taichi_aot.capabilities import classify_device
from taichi_vision.taichi_aot.gfx_capabilities import (
    negotiate_graphics_capabilities,
)


def test_missing_vulkan_probe_evidence_is_unknown() -> None:
    snapshot = negotiate_graphics_capabilities("vulkan", {"name": "RTX"})
    assert snapshot.usable is False
    assert snapshot.decision.status == "unknown"


def test_vulkan_snapshot_requires_and_preserves_probe_evidence() -> None:
    snapshot = negotiate_graphics_capabilities(
        "vulkan",
        {
            "api_version": "1.3",
            "features": {"compute": True, "ssbo": True},
            "capability_source": "vulkan-probe",
        },
    )
    assert snapshot.usable is True
    assert snapshot.evidence_source == "vulkan-probe"
    assert snapshot.features == frozenset({"compute", "ssbo"})


def test_backend_admission_does_not_promote_vulkan_vendor_name() -> None:
    missing = classify_device(
        {"name": "NVIDIA RTX Test", "vendor": "NVIDIA"}, "vulkan"
    )
    qualified = classify_device(
        {
            "name": "NVIDIA RTX Test",
            "vendor": "NVIDIA",
            "api_version": "1.3",
            "features": {"compute": True, "ssbo": True},
            "capability_source": "vulkan-probe",
        },
        "vulkan",
    )
    assert missing.safe is False
    assert qualified.safe is True


def test_opengl_snapshot_rejects_missing_or_legacy_profile() -> None:
    missing = negotiate_graphics_capabilities("opengl", {"name": "GPU"})
    legacy = negotiate_graphics_capabilities(
        "opengl",
        {
            "api_version": "4.2",
            "compute_shader": True,
            "ssbo": True,
            "capability_source": "gl-probe",
        },
    )
    qualified = negotiate_graphics_capabilities(
        "opengl",
        {
            "api_version": "4.3",
            "compute_shader": True,
            "ssbo": True,
            "capability_source": "gl-probe",
        },
    )
    assert missing.decision.status == "unknown"
    assert legacy.usable is False
    assert qualified.usable is True


def test_environment_override_is_not_production_evidence(monkeypatch) -> None:
    evidence = {
        "api_version": "1.3",
        "features": {"compute": True, "ssbo": True},
        "capability_source": "environment_override",
    }
    monkeypatch.delenv("AOT_CAPABILITY_OVERRIDE", raising=False)
    rejected = negotiate_graphics_capabilities("vulkan", evidence)
    monkeypatch.setenv("AOT_CAPABILITY_OVERRIDE", "1")
    accepted = negotiate_graphics_capabilities("vulkan", evidence)

    assert rejected.usable is False
    assert accepted.usable is True
    assert accepted.explicit_override is True


def test_engine_does_not_fabricate_graphics_feature_bits() -> None:
    source = (Path(__file__).parents[1] / "engine.py").read_text(encoding="utf-8")
    assert 'runtime_features = {"COMPUTE", "SSBO"}' not in source
    assert "AOT_CAPABILITY_OVERRIDE" in source
    assert "_graphics_capability_snapshot" in source
