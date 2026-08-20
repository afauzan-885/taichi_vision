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


def test_graph_and_pipeline_replay_reject_cross_runtime_or_stale_modules():
    assert '"run_aot_graph: module handle is stale"' in SOURCE
    assert '"run_aot_graph: module is not live"' in SOURCE
    assert '"run_aot_graph: module belongs to a different runtime"' in SOURCE
    assert '"run_pipeline: module belongs to a different runtime"' in SOURCE
    assert '"run_pipeline: module handle is stale"' in SOURCE
    assert "ctx->owner != engine" in SOURCE


def test_module_retirement_invalidates_owner_pipeline_steps_before_delete():
    assert "invalidate_pipelines_for_module" in SOURCE
    destroy_start = SOURCE.index("static ModuleContext *begin_module_destroy")
    destroy_end = SOURCE.index("static void finish_module_destroy", destroy_start)
    destroy_source = SOURCE[destroy_start:destroy_end]
    assert "ctx->destroying = true" in destroy_source
    assert "invalidate_pipelines_for_module(ctx)" in destroy_source
    invalidate_start = SOURCE.index("static void invalidate_pipelines_for_module")
    invalidate_end = SOURCE.index("#ifdef _WIN32", invalidate_start)
    invalidate_source = SOURCE[invalidate_start:invalidate_end]
    assert "step.module_ctx == ctx" in invalidate_source
    assert "owner->pipelines.erase" in invalidate_source
