"""Regression contracts for transactional native GPU image I/O."""

from __future__ import annotations

from pathlib import Path


ENGINE_SOURCE = (
    Path(__file__).parents[2] / "taichi_algorithm" / "aot_py" / "taichi_aot_engine.cpp"
).read_text(encoding="utf-8")


def _function_body(name: str) -> str:
    start = ENGINE_SOURCE.index(f"EXPORT {name}")
    end = ENGINE_SOURCE.find("EXPORT ", start + len("EXPORT "))
    return ENGINE_SOURCE[start : end if end >= 0 else len(ENGINE_SOURCE)]


def test_writer_maps_before_creating_destination_stream():
    body = _function_body("bool ti_imwrite_from_gpu")
    map_pos = body.index("ti_map_memory")
    stream_pos = body.index("CreateStream")

    assert map_pos < stream_pos
    assert "if (!gpu_ptr)" in body


def test_writer_unmaps_source_on_every_post_map_failure_path():
    body = _function_body("bool ti_imwrite_from_gpu")

    assert "auto cleanup = [&]()" in body
    cleanup = body[body.index("auto cleanup = [&]()") :]
    assert "ti_unmap_memory" in cleanup
    assert "gpu_ptr = nullptr" in cleanup
    assert "FAILED(write_result)" in body


def test_reader_exposes_handle_only_after_checked_pixel_copy():
    body = _function_body("void *ti_imread_to_gpu")
    return_pos = body.rindex("return (void *)gpu_mem")
    copy_check_pos = body.index("if (!copy_ok)")

    assert copy_check_pos < return_pos
    assert "ti_free_memory" in body[copy_check_pos:return_pos]
