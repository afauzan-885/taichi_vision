"""Source-level guard for native EngineContext admission and lifetime ordering."""

from __future__ import annotations

from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1] / "taichi_aot_engine.cpp"
).read_text(encoding="utf-8")


def test_native_bridge_has_context_lease_admission():
    assert "class EngineLease" in SOURCE
    assert "candidate->active_calls++" in SOURCE
    assert "candidate->destroying" in SOURCE
    assert "lifecycle_cv.wait" in SOURCE


def test_destroy_closes_admission_before_graphics_rebind():
    destroy_start = SOURCE.index("EXPORT void destroy_aot_engine")
    destroy_source = SOURCE[destroy_start:]
    assert destroy_source.index("engine_contexts.erase") < destroy_source.index(
        "ScopedOpenGLContext gl_scope"
    )
    assert "active_calls == 0" in destroy_source
