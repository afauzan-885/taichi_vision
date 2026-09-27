"""Build only the resident SR preparation graphs, one target per process."""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ["AOT_MODE"] = "0"
import taichi as ti


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compile_sr_resident(arch=None, save_path=None):
    """Compile the shared graphs using the backend suite's path convention."""
    arch = ti.cpu if arch is None else arch
    ti.init(arch=arch, offline_cache=False)
    root = Path(__file__).resolve().parent
    module = ti.aot.Module(arch)
    load_source("sr_preparation_kernels", root / "sr_resident.py").build_graphs(module)
    module.archive(str(save_path))
    load_source("accumulator_artifact", root / "aot_py/aot_artifact.py").normalize_tcm(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("cpu", "cuda", "vulkan", "opengl"))
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    arch = getattr(ti, args.backend)
    target_module = load_source("accumulator_targets", root.parent / "taichi_aot/artifact_targets.py")
    target = target_module.detect_target(backend=args.backend, device="nvidia" if args.backend == "cuda" else "")
    out = Path(args.output_root) / target.target_id / f"sr_resident_{target.target_id}.tcm"
    out.parent.mkdir(parents=True, exist_ok=True)
    compile_sr_resident(arch=arch, save_path=out)
    print(f"[AOT] backend={args.backend} target={target.target_id} artifact={out}")
    ti.reset()
