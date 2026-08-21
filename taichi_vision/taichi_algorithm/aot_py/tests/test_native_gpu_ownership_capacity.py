"""Focused native admission checks for issue #24.

The compiled smoke is intentionally optional: source-only CI still exercises
the ordering and admission contract, while a locally built CPU bridge proves
the checks occur before the Taichi C API is called.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
CPP = (Path(__file__).resolve().parents[1] / "taichi_aot_engine.cpp").read_text(
    encoding="utf-8"
)
BRIDGE = Path(
    os.environ.get(
        "PIXEL_REFINE_TEST_BRIDGE",
        str(
            ROOT
            / "taichi_vision"
            / "taichi_algorithm"
            / "aot_py"
            / "taichi_aot_engine.dll"
        ),
    )
)
LLVM_C_API = Path(
    os.environ.get(
        "PIXEL_REFINE_TAICHI_C_API_DIR",
        r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\c_api\bin",
    )
)


def _body(start: str, end: str) -> str:
    first = CPP.index(start)
    return CPP[first : CPP.index(end, first)]


def test_native_allocation_admission_rejects_zero_and_reports_map_failures():
    allocation = _body("EXPORT void *allocate_gpu_buffer", "EXPORT void free_gpu_buffer")
    write = _body("EXPORT void write_to_gpu_buffer", "EXPORT void read_from_gpu_buffer")
    read = _body("EXPORT void read_from_gpu_buffer", "EXPORT void *map_gpu_buffer")
    assert "size == 0" in allocation
    assert "allocation size must be positive" in allocation
    assert "GPU map failed" in write
    assert "GPU map failed" in read


def test_native_graph_admission_bounds_counts_before_vector_reserve():
    graph = _body("EXPORT void run_aot_graph", "// -----------------------------------------------------------------------\n// Pipeline Recording")
    add = _body("EXPORT void add_to_pipeline", "EXPORT void run_pipeline")
    pipeline = _body("EXPORT void run_pipeline", "} // extern \"C\"")
    for body, reserve in ((graph, "ti_args.reserve(num_args)"), (add, "dispatch.args.reserve(num_args)")):
        assert "num_args < 0 || num_args > kMaxDynamicArgs" in body
        assert body.index("num_args < 0 || num_args > kMaxDynamicArgs") < body.index(reserve)
    assert "num_overrides < 0 || num_overrides > kMaxDynamicArgs" in pipeline
    assert "override arrays are null" in pipeline


def test_dynamic_arg_shape_is_validated_before_allocation_capacity_calculation():
    fill = _body("static bool _fill_ti_arg", "// -----------------------------------------------------------------------\n// Generic Graph Execution")
    assert fill.index("dyn_arg.shape[d] <= 0") < fill.index(
        "validate_dynamic_arg_allocation"
    )
    assert "ndarray element-count overflow" in CPP
    assert "transfer exceeds GPU allocation capacity" in CPP


@pytest.mark.skipif(os.name != "nt", reason="compiled bridge smoke is Windows-specific")
@pytest.mark.skipif(not BRIDGE.exists(), reason="compiled CPU bridge is not built")
@pytest.mark.skipif(
    not (LLVM_C_API / "taichi_c_api.dll").exists(),
    reason="LLVM20 C API runtime is unavailable",
)
def test_compiled_native_issue24_admission_is_fail_closed():
    code = r'''
import ctypes, os
from pathlib import Path

bridge = Path(os.environ["PIXEL_REFINE_TEST_BRIDGE"]).resolve()
runtime_bin = Path(os.environ["PIXEL_REFINE_TAICHI_C_API_DIR"]).resolve()
os.environ["TI_LIB_DIR"] = str(runtime_bin.parent.parent / "runtime")
os.add_dll_directory(str(runtime_bin))
os.add_dll_directory(str(bridge.parent))
d = ctypes.WinDLL(str(bridge))
d.init_aot_engine.argtypes = [ctypes.c_int, ctypes.c_int]
d.init_aot_engine.restype = ctypes.c_void_p
d.destroy_aot_engine.argtypes = [ctypes.c_void_p]
d.allocate_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
d.allocate_gpu_buffer.restype = ctypes.c_void_p
d.free_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
d.write_to_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
d.read_from_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
d.copy_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
d.map_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
d.map_gpu_buffer.restype = ctypes.c_void_p
d.unmap_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
d.get_last_engine_error.argtypes = [ctypes.c_void_p]
d.get_last_engine_error.restype = ctypes.c_char_p

def error(runtime):
    return (d.get_last_engine_error(runtime) or b"").lower()

r1 = d.init_aot_engine(0, 0)
r2 = d.init_aot_engine(0, 0)
assert r1 and r2 and r1 != r2
assert not d.allocate_gpu_buffer(r1, 0, 1)
assert b"positive" in error(r1)
a = d.allocate_gpu_buffer(r1, 16, 1)
b = d.allocate_gpu_buffer(r1, 8, 1)
foreign = d.allocate_gpu_buffer(r2, 16, 1)
assert a and b and foreign
payload = (ctypes.c_ubyte * 16)(*range(16))
out = (ctypes.c_ubyte * 16)()
d.write_to_gpu_buffer(r1, a, payload, 17)
assert b"capacity" in error(r1)
d.read_from_gpu_buffer(r1, a, out, 17)
assert b"capacity" in error(r1)
d.copy_gpu_buffer(r1, a, b, 16)
assert b"capacity" in error(r1)
d.copy_gpu_buffer(r1, a, foreign, 16)
assert b"does not belong" in error(r1)
assert d.map_gpu_buffer(r1, a)
assert not d.map_gpu_buffer(r1, a)
assert b"already mapped" in error(r1)
d.free_gpu_buffer(r1, a)
assert b"already mapped" in error(r1)
d.unmap_gpu_buffer(r1, a)
d.free_gpu_buffer(r1, a)
d.free_gpu_buffer(r1, a)
assert b"does not belong" in error(r1)
d.free_gpu_buffer(r1, b)
d.free_gpu_buffer(r2, foreign)
d.destroy_aot_engine(r1)
d.destroy_aot_engine(r2)
os._exit(0)
'''
    env = os.environ.copy()
    env["PIXEL_REFINE_TEST_BRIDGE"] = str(BRIDGE)
    env["PIXEL_REFINE_TAICHI_C_API_DIR"] = str(LLVM_C_API)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
