"""Regression guards for the native GPU allocation ownership contract."""

from __future__ import annotations

from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1] / "taichi_aot_engine.cpp"
).read_text(encoding="utf-8")


def _function_source(name: str, next_name: str) -> str:
    start = SOURCE.index(f"EXPORT {name}")
    end = SOURCE.index(f"EXPORT {next_name}", start)
    return SOURCE[start:end]


def test_native_context_records_capacity_and_mapping_state():
    assert "struct GpuAllocationRecord" in SOURCE
    assert "uint64_t size = 0" in SOURCE
    assert "bool mapped = false" in SOURCE
    assert "std::unordered_map<TiMemory, GpuAllocationRecord> allocations" in SOURCE
    assert "std::unordered_set<TiMemory> buffers" not in SOURCE
    assert "std::unordered_set<TiMemory> mapped_buffers" not in SOURCE


def test_every_direct_memory_entry_point_uses_the_shared_validator():
    for name, next_name in (
        ("void free_gpu_buffer", "void write_to_gpu_buffer"),
        ("void write_to_gpu_buffer", "void read_from_gpu_buffer"),
        ("void read_from_gpu_buffer", "void *map_gpu_buffer"),
        ("void *map_gpu_buffer", "void unmap_gpu_buffer"),
        ("void unmap_gpu_buffer", "void copy_gpu_buffer"),
        ("void copy_gpu_buffer", "void sync_runtime"),
    ):
        body = _function_source(name, next_name)
        assert "validate_gpu_allocation_locked" in body, name


def test_range_validation_precedes_native_memory_access():
    for name, next_name, native_call in (
        ("void write_to_gpu_buffer", "void read_from_gpu_buffer", "ti_map_memory"),
        ("void read_from_gpu_buffer", "void *map_gpu_buffer", "ti_map_memory"),
        ("void copy_gpu_buffer", "void sync_runtime", "ti_copy_memory_device_to_device"),
    ):
        body = _function_source(name, next_name)
        assert body.index("validate_gpu_allocation_locked") < body.index(native_call)


def test_image_io_uses_the_same_allocation_contract():
    read_body = _function_source("void *ti_imread_to_gpu", "bool ti_imwrite_from_gpu")
    write_body = _function_source("bool ti_imwrite_from_gpu", "bool ti_cast_buffer")
    assert "allocations.emplace" in read_body
    assert "allocation_it->second.mapped = true" in read_body
    assert "allocation_it->second.mapped = false" in read_body
    assert "validate_gpu_allocation_locked" in write_body
    assert "size64" in write_body

