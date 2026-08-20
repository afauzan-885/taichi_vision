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


def test_native_pipeline_module_handles_are_leased():
    assert "class ModuleLease" in SOURCE
    assert "module_contexts" in SOURCE
    assert "ModuleLease module_lease(module_ctx)" in SOURCE
    assert "ModuleLease module_lease(step.module_ctx)" in SOURCE
    assert "begin_module_destroy" in SOURCE
    assert "finish_module_destroy" in SOURCE
