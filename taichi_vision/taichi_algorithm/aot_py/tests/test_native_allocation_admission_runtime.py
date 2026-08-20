"""Optional compiled-bridge admission smoke for the Windows CPU bundle."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
BRIDGE = ROOT / "taichi_vision" / "taichi_algorithm" / "aot_py" / "taichi_aot_engine.dll"
LLVM_BIN = Path(
    r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\c_api\bin"
)


@pytest.mark.skipif(os.name != "nt", reason="compiled bridge smoke is Windows-specific")
@pytest.mark.skipif(not BRIDGE.exists(), reason="publish CPU bridge is not built")
@pytest.mark.skipif(
    not (LLVM_BIN / "taichi_c_api.dll").exists(),
    reason="LLVM20 C API runtime is unavailable",
)
def test_compiled_bridge_rejects_foreign_oversized_and_double_map_operations():
    code = r'''
import ctypes, os
from pathlib import Path
os.add_dll_directory(r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\c_api\bin")
os.environ["TI_LIB_DIR"] = r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\runtime"
d = ctypes.WinDLL(str(Path(r"taichi_vision/taichi_algorithm/aot_py/taichi_aot_engine.dll").resolve()))
d.init_aot_engine.argtypes = [ctypes.c_int, ctypes.c_int]
d.init_aot_engine.restype = ctypes.c_void_p
d.get_runtime_arch_id.argtypes = [ctypes.c_void_p]
d.get_runtime_arch_id.restype = ctypes.c_int
d.allocate_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
d.allocate_gpu_buffer.restype = ctypes.c_void_p
d.write_to_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
d.free_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
d.map_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
d.map_gpu_buffer.restype = ctypes.c_void_p
d.unmap_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
d.get_last_engine_error.argtypes = [ctypes.c_void_p]
d.get_last_engine_error.restype = ctypes.c_char_p
r1 = d.init_aot_engine(0, 0)
r2 = d.init_aot_engine(0, 1)
assert r1 and r2
assert d.get_runtime_arch_id(r1) == 0
assert d.get_runtime_arch_id(r2) == 0
b1 = d.allocate_gpu_buffer(r1, 16, 1)
b2 = d.allocate_gpu_buffer(r2, 16, 1)
assert b1 and b2
payload = (ctypes.c_ubyte * 16)(*range(16))
d.write_to_gpu_buffer(r1, b1, payload, 17)
assert b"capacity" in (d.get_last_engine_error(r1) or b"")
assert d.map_gpu_buffer(r1, b1)
d.map_gpu_buffer(r1, b1)
assert b"already mapped" in (d.get_last_engine_error(r1) or b"")
d.unmap_gpu_buffer(r1, b1)
d.free_gpu_buffer(r2, b1)
assert b"does not belong" in (d.get_last_engine_error(r2) or b"")
os._exit(0)
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


@pytest.mark.skipif(os.name != "nt", reason="compiled bridge smoke is Windows-specific")
@pytest.mark.skipif(not BRIDGE.exists(), reason="publish CPU bridge is not built")
@pytest.mark.skipif(
    not (LLVM_BIN / "taichi_c_api.dll").exists(),
    reason="LLVM20 C API runtime is unavailable",
)
def test_compiled_bridge_wic_write_roundtrip_and_geometry_fail_closed():
    code = r'''
import ctypes, os, tempfile
from pathlib import Path
os.add_dll_directory(r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\c_api\bin")
os.environ["TI_LIB_DIR"] = r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\runtime"
d = ctypes.WinDLL(str(Path(r"taichi_vision/taichi_algorithm/aot_py/taichi_aot_engine.dll").resolve()))
d.init_aot_engine.argtypes = [ctypes.c_int, ctypes.c_int]
d.init_aot_engine.restype = ctypes.c_void_p
d.allocate_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
d.allocate_gpu_buffer.restype = ctypes.c_void_p
d.write_to_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
d.ti_imwrite_from_gpu.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
d.ti_imwrite_from_gpu.restype = ctypes.c_bool
d.ti_imread_to_gpu.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
d.ti_imread_to_gpu.restype = ctypes.c_void_p
d.free_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
d.get_last_engine_error.argtypes = [ctypes.c_void_p]
d.get_last_engine_error.restype = ctypes.c_char_p
r = d.init_aot_engine(2, 0)
b = d.allocate_gpu_buffer(r, 4 * 4 * 3, 1)
payload = (ctypes.c_ubyte * (4 * 4 * 3))(*([128] * (4 * 4 * 3)))
d.write_to_gpu_buffer(r, b, payload, len(payload))
out = Path(tempfile.gettempdir()) / "taichi_vision_native_roundtrip.png"
if out.exists(): out.unlink()
assert d.ti_imwrite_from_gpu(r, str(out).encode(), b, 4, 4, 3, 8)
assert out.exists() and out.stat().st_size > 0
for stage, expected in (("read_map", "map"), ("read_copy", "copy")):
    os.environ["PIXEL_REFINE_AOT_TEST_FAIL_IMAGE_IO"] = stage
    rw, rh, rc, rd = (ctypes.c_int() for _ in range(4))
    assert not d.ti_imread_to_gpu(r, str(out).encode(), ctypes.byref(rw), ctypes.byref(rh), ctypes.byref(rc), ctypes.byref(rd))
    assert expected.encode() in (d.get_last_engine_error(r) or b"")
os.environ.pop("PIXEL_REFINE_AOT_TEST_FAIL_IMAGE_IO", None)
rw, rh, rc, rd = (ctypes.c_int() for _ in range(4))
read_handle = d.ti_imread_to_gpu(r, str(out).encode(), ctypes.byref(rw), ctypes.byref(rh), ctypes.byref(rc), ctypes.byref(rd))
assert read_handle and (rw.value, rh.value, rc.value, rd.value) == (4, 4, 3, 8)
d.free_gpu_buffer(r, read_handle)
sentinel = Path(tempfile.gettempdir()) / "taichi_vision_native_sentinel.png"
sentinel.write_bytes(b"keep")
assert not d.ti_imwrite_from_gpu(r, str(sentinel).encode(), b, 0, 4, 3, 8)
assert sentinel.read_bytes() == b"keep"
for stage, expected in (("map", "map"), ("encoder", "encoder"), ("frame_commit", "frame commit"), ("encoder_commit", "encoder commit"), ("replace", "replace")):
    os.environ["PIXEL_REFINE_AOT_TEST_FAIL_IMAGE_IO"] = stage
    assert not d.ti_imwrite_from_gpu(r, str(sentinel).encode(), b, 4, 4, 3, 8)
    assert expected.encode() in (d.get_last_engine_error(r) or b"")
    assert sentinel.read_bytes() == b"keep", (stage, sentinel.read_bytes())
os._exit(0)
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


@pytest.mark.skipif(os.name != "nt", reason="compiled bridge smoke is Windows-specific")
@pytest.mark.skipif(
    not (LLVM_BIN / "taichi_c_api.dll").exists(),
    reason="LLVM20 C API runtime is unavailable",
)
def test_all_packaged_windows_bridges_export_runtime_identity_probe():
    code = r'''
import ctypes
from pathlib import Path
import os
root = Path(r"taichi_vision/taichi_algorithm/aot_py/aot_dll")
for target in ("cpu", "cuda", "opengl", "vulkan"):
    target_dir = (root / target).resolve()
    os.add_dll_directory(str(target_dir))
    bridge = ctypes.WinDLL(str((target_dir / "taichi_aot_engine.dll").resolve()))
    assert hasattr(bridge, "get_runtime_arch_id"), target
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
