"""Batch compiler for compression TCM modules across all supported backends."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set compile-only flag BEFORE importing any AOT modules
os.environ["PIXEL_REFINE_AOT_COMPILE_ONLY"] = "1"

import taichi as ti
from taichi_vision.taichi_algorithm.compression.compile_compression_image_tcm import compile_compression

def compile_all():
    targets = [
        ("cpu_x86_64_windows", ti.cpu),
        ("opengl_x86_64_windows", ti.opengl),
    ]
    for target_name, arch in targets:
        print(f"\n==========================================")
        print(f"Compiling compression_image TCM for: {target_name} ({arch})")
        print(f"==========================================")
        os.environ["PIXEL_REFINE_TARGET_VARIANT"] = target_name
        try:
            out = compile_compression(arch=arch)
            print(f"Successfully compiled: {out}")
        except Exception as e:
            print(f"Warning: could not compile {target_name}: {e}")

if __name__ == "__main__":
    compile_all()
