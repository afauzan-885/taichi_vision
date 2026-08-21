"""Python-side GPU wrapper ownership gates."""

from __future__ import annotations

from pathlib import Path


SOURCE = (Path(__file__).parents[1] / "engine.py").read_text(encoding="utf-8")


def test_python_owner_validator_is_shared_by_all_handle_boundaries():
    assert "def _validate_gpu_buffer_owner" in SOURCE
    assert "different AOT runtime" in SOURCE
    assert "no owning AOT runtime" in SOURCE
    assert "_validate_gpu_buffer_owner(data, self, \"upload\")" in SOURCE
    assert "_validate_gpu_buffer_owner(buf, self, \"imwrite\")" in SOURCE


def test_graph_and_pipeline_dispatch_supply_the_expected_engine():
    assert "expected_engine=engine" in SOURCE
    assert "expected_engine=self" in SOURCE
    assert "Pipeline '{name}' override key" in SOURCE
