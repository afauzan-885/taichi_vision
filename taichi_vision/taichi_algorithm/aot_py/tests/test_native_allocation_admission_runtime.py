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
def test_compiled_graph_rejects_stale_module_handle_cleanly():
    code = r'''
import ctypes, importlib.util, os, sys
from pathlib import Path
os.environ["TI_LIB_DIR"] = r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\runtime"
os.add_dll_directory(r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\c_api\bin")
source = Path(r"taichi_vision/taichi_aot/engine.py").resolve()
spec = importlib.util.spec_from_file_location("taichi_vision.taichi_aot.stale_module_probe", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module._WATCHDOG_STOP.set()
if getattr(module, "_watchdog", None) is not None:
    module._watchdog.join(timeout=2)
tcm = next(Path(r"taichi_vision/taichi_algorithm/aot_tcm/cpu_x86_64_windows").glob("bilinear_demosaice*.tcm"))
module_ptr = module._LIB.load_aot_module(module.engine.runtime, str(tcm).encode())
assert module_ptr
module._LIB.destroy_aot_module(module_ptr)
module._LIB.run_aot_graph(module.engine.runtime, module_ptr, b"pure_bilinear_demosaice", None, 0)
assert "stale" in (module._get_native_engine_error(module.engine.runtime) or "")
os._exit(0)
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


@pytest.mark.skipif(os.name != "nt", reason="compiled bridge smoke is Windows-specific")
@pytest.mark.skipif(not BRIDGE.exists(), reason="publish CPU bridge is not built")
@pytest.mark.skipif(
    not (LLVM_BIN / "taichi_c_api.dll").exists(),
    reason="LLVM20 C API runtime is unavailable",
)
def test_compiled_graph_rejects_foreign_runtime_dynamic_arg_before_launch():
    code = r'''
import ctypes, importlib.util, os, sys
from pathlib import Path
os.environ["TI_LIB_DIR"] = r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\runtime"
os.add_dll_directory(r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\c_api\bin")
source = Path(r"taichi_vision/taichi_aot/engine.py").resolve()
spec = importlib.util.spec_from_file_location("taichi_vision.taichi_aot.graph_admission_probe", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module._WATCHDOG_STOP.set()
if getattr(module, "_watchdog", None) is not None:
    module._watchdog.join(timeout=2)
tcm = next(Path(r"taichi_vision/taichi_algorithm/aot_tcm/cpu_x86_64_windows").glob("bilinear_demosaice*.tcm"))
module._LIB.load_aot_module.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
module._LIB.load_aot_module.restype = ctypes.c_void_p
module._LIB.init_aot_engine.argtypes = [ctypes.c_int, ctypes.c_int]
module._LIB.init_aot_engine.restype = ctypes.c_void_p
module._LIB.allocate_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
module._LIB.allocate_gpu_buffer.restype = ctypes.c_void_p
owner = module.engine.runtime
module_ptr = module._LIB.load_aot_module(owner, str(tcm).encode())
foreign_runtime = module._LIB.init_aot_engine(0, 0)
foreign_handle = module._LIB.allocate_gpu_buffer(foreign_runtime, 64, 1)
assert module_ptr and foreign_runtime and foreign_handle
arg = module.DynamicArg()
arg.name = b"bayer"
arg.arg_type = 0
arg.dtype = 0
arg.dim_count = 2
arg.shape[0] = 4
arg.shape[1] = 4
arg.val_u64 = foreign_handle
args = (module.DynamicArg * 1)(arg)
module._LIB.run_aot_graph(owner, module_ptr, b"pure_bilinear_demosaice", args, 1)
assert "does not belong" in (module._get_native_engine_error(owner) or "")
os._exit(0)
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
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
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]
psapi = ctypes.WinDLL("psapi.dll")
kernel32 = ctypes.WinDLL("kernel32.dll")
psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_uint32]
psapi.GetProcessMemoryInfo.restype = ctypes.c_int
kernel32.GetCurrentProcess.restype = ctypes.c_void_p
def private_bytes():
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    assert psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return counters.PrivateUsage
private_before = private_bytes()
for stage, expected in (("read_map", "map"), ("read_copy", "copy")):
    os.environ["PIXEL_REFINE_AOT_TEST_FAIL_IMAGE_IO"] = stage
    for _ in range(24):
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
    for _ in range(24):
        assert not d.ti_imwrite_from_gpu(r, str(sentinel).encode(), b, 4, 4, 3, 8)
        assert expected.encode() in (d.get_last_engine_error(r) or b"")
        assert sentinel.read_bytes() == b"keep", (stage, sentinel.read_bytes())
private_after = private_bytes()
assert private_after - private_before < 128 * 1024 * 1024, (private_before, private_after)
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


@pytest.mark.skipif(os.name != "nt", reason="compiled bridge smoke is Windows-specific")
@pytest.mark.skipif(not BRIDGE.exists(), reason="publish CPU bridge is not built")
@pytest.mark.skipif(
    not (LLVM_BIN / "taichi_c_api.dll").exists(),
    reason="LLVM20 C API runtime is unavailable",
)
def test_forced_gpu_init_failure_reports_explicit_cpu_fallback_identity():
    code = r'''
import ctypes, os
from pathlib import Path
os.environ["PIXEL_REFINE_AOT_TEST_FAIL_INIT"] = "1"
os.environ["PIXEL_REFINE_AOT_ALLOW_CPU_FALLBACK"] = "1"
os.add_dll_directory(r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\c_api\bin")
os.environ["TI_LIB_DIR"] = r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\runtime"
d = ctypes.WinDLL(str(Path(r"taichi_vision/taichi_algorithm/aot_py/taichi_aot_engine.dll").resolve()))
d.init_aot_engine.argtypes = [ctypes.c_int, ctypes.c_int]
d.init_aot_engine.restype = ctypes.c_void_p
d.get_runtime_arch_id.argtypes = [ctypes.c_void_p]
d.get_runtime_arch_id.restype = ctypes.c_int
d.get_runtime_device_name.argtypes = [ctypes.c_void_p]
d.get_runtime_device_name.restype = ctypes.c_char_p
d.get_last_engine_error.argtypes = [ctypes.c_void_p]
d.get_last_engine_error.restype = ctypes.c_char_p
runtime = d.init_aot_engine(1, 0)
assert runtime
assert d.get_runtime_arch_id(runtime) == 2
assert b"CPU" in (d.get_runtime_device_name(runtime) or b"")
assert b"CPU fallback" in (d.get_last_engine_error(runtime) or b"")
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
def test_compiled_same_name_pipeline_clear_is_scoped_to_each_runtime():
    code = r'''
import ctypes, os
from pathlib import Path
os.add_dll_directory(r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\c_api\bin")
os.environ["TI_LIB_DIR"] = r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\runtime"
d = ctypes.WinDLL(str(Path(r"taichi_vision/taichi_algorithm/aot_py/taichi_aot_engine.dll").resolve()))
class DynamicArg(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("arg_type", ctypes.c_int), ("dtype", ctypes.c_int), ("dim_count", ctypes.c_int), ("shape", ctypes.c_int * 8), ("elem_dim_count", ctypes.c_int), ("elem_shape", ctypes.c_int * 8), ("is_vector", ctypes.c_int), ("vector_dim", ctypes.c_int), ("val_u64", ctypes.c_uint64)]
d.init_aot_engine.argtypes = [ctypes.c_int, ctypes.c_int]
d.init_aot_engine.restype = ctypes.c_void_p
d.load_aot_module.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
d.load_aot_module.restype = ctypes.c_void_p
d.allocate_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
d.allocate_gpu_buffer.restype = ctypes.c_void_p
d.clear_pipeline_for_engine.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
d.add_to_pipeline.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(DynamicArg), ctypes.c_int]
d.run_pipeline.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(DynamicArg), ctypes.c_int]
d.get_last_engine_error.argtypes = [ctypes.c_void_p]
d.get_last_engine_error.restype = ctypes.c_char_p
runtime_a = d.init_aot_engine(2, 0)
runtime_b = d.init_aot_engine(2, 0)
assert runtime_a and runtime_b and runtime_a != runtime_b
tcm = next(Path(r"taichi_vision/taichi_algorithm/aot_tcm/cpu_x86_64_windows").glob("bilinear_demosaice*.tcm"))
module_a = d.load_aot_module(runtime_a, str(tcm).encode())
module_b = d.load_aot_module(runtime_b, str(tcm).encode())
assert module_a and module_b
def make_args(runtime):
    bayer = d.allocate_gpu_buffer(runtime, 4 * 4 * 4, 1)
    dst = d.allocate_gpu_buffer(runtime, 4 * 4 * 3 * 4, 1)
    assert bayer and dst
    specs = [(b"bayer", 0, 0, 2, bayer), (b"dst", 0, 0, 3, dst), (b"black", 1, 0, 0, 0), (b"white", 1, 0, 0, 65535), (b"h", 1, 1, 0, 4), (b"w", 1, 1, 0, 4), (b"c00", 1, 1, 0, 0), (b"c01", 1, 1, 0, 1), (b"c10", 1, 1, 0, 3), (b"c11", 1, 1, 0, 2)]
    args = (DynamicArg * len(specs))()
    for i, (name, arg_type, dtype, dim_count, value) in enumerate(specs):
        args[i].name = name
        args[i].arg_type = arg_type
        args[i].dtype = dtype
        args[i].dim_count = dim_count
        if arg_type == 0:
            args[i].shape[0] = 4
            args[i].shape[1] = 4
            if dim_count == 3:
                args[i].shape[2] = 3
        args[i].val_u64 = value
    return args
args_a = make_args(runtime_a)
args_b = make_args(runtime_b)
d.add_to_pipeline(module_a, b"same-name", b"pure_bilinear_demosaice", args_a, 10)
d.add_to_pipeline(module_b, b"same-name", b"pure_bilinear_demosaice", args_b, 10)
d.clear_pipeline_for_engine(runtime_a, b"same-name")
d.run_pipeline(runtime_b, b"same-name", None, None, 0)
assert not (d.get_last_engine_error(runtime_b) or b"")
os._exit(0)
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


@pytest.mark.skipif(os.name != "nt", reason="compiled bridge smoke is Windows-specific")
@pytest.mark.skipif(not BRIDGE.exists(), reason="publish CPU bridge is not built")
@pytest.mark.skipif(
    not (LLVM_BIN / "taichi_c_api.dll").exists(),
    reason="LLVM20 C API runtime is unavailable",
)
def test_compiled_repeated_module_destroy_and_stale_replay_is_safe():
    code = r'''
import ctypes, os
from pathlib import Path
os.add_dll_directory(r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\c_api\bin")
os.environ["TI_LIB_DIR"] = r"D:\development_build\taichi_runtime_llvm20\release-runtime\taichi\_lib\runtime"
d = ctypes.WinDLL(str(Path(r"taichi_vision/taichi_algorithm/aot_py/taichi_aot_engine.dll").resolve()))
d.init_aot_engine.argtypes = [ctypes.c_int, ctypes.c_int]
d.init_aot_engine.restype = ctypes.c_void_p
d.load_aot_module.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
d.load_aot_module.restype = ctypes.c_void_p
d.destroy_aot_module.argtypes = [ctypes.c_void_p]
d.run_aot_graph.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int]
d.get_last_engine_error.argtypes = [ctypes.c_void_p]
d.get_last_engine_error.restype = ctypes.c_char_p
runtime = d.init_aot_engine(2, 0)
assert runtime
tcm = next(Path(r"taichi_vision/taichi_algorithm/aot_tcm/cpu_x86_64_windows").glob("bilinear_demosaice*.tcm"))
for _ in range(24):
    module_ptr = d.load_aot_module(runtime, str(tcm).encode())
    assert module_ptr
    d.destroy_aot_module(module_ptr)
    d.run_aot_graph(runtime, module_ptr, b"pure_bilinear_demosaice", None, 0)
    assert b"stale" in (d.get_last_engine_error(runtime) or b"")
os._exit(0)
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
