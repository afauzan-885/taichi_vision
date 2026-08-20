"""Source contract for owner-scoped Python pipeline cleanup."""

from __future__ import annotations

from pathlib import Path


SOURCE = (Path(__file__).parents[1] / "engine.py").read_text(encoding="utf-8")


def test_normal_pipeline_cleanup_routes_through_module_owner():
    start = SOURCE.index("def _drop_pipeline_recording")
    end = SOURCE.index("def _abort_auto_pipeline", start)
    block = SOURCE[start:end]
    assert "_LIB.clear_pipeline(module_ptr" in block
    assert "owner_count == 0" in block


def test_clear_pipeline_by_name_does_not_broadcast_when_modules_exist():
    start = SOURCE.index("def clear_pipeline_by_name")
    end = SOURCE.index("def clear_pipelines", start)
    block = SOURCE[start:end]
    assert "_LIB.clear_pipeline(module_ptr" in block
    assert "if owner_count == 0" in block


def test_engine_destroy_clears_pipelines_before_unloading_modules():
    start = SOURCE.index("def destroy(self):")
    end = SOURCE.index("def _global_cleanup", start)
    block = SOURCE[start:end]
    assert block.index("self.clear_pipelines()") < block.index(
        "# 5. Unload all AOT modules"
    )
