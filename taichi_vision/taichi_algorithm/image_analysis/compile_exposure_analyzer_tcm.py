"""Compile the target-qualified Taichi AOT exposure histogram graph."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("AOT_MODE", "0")

import taichi as ti

from taichi_vision.taichi_aot.artifact_targets import detect_target
from taichi_vision.taichi_algorithm.image_analysis.exposure_analyzer import (
    EXPOSURE_HISTOGRAM_BINS,
    EXPOSURE_SAMPLE_CAP,
    _exposure_histogram_f32,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "taichi_vision" / "taichi_algorithm" / "aot_tcm"


def compile_exposure_analyzer(backend: str, vendor: str | None = None) -> Path:
    target = detect_target(backend=backend, device=vendor)
    arch_by_backend = {
        "cpu": ti.cpu,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "opengl": ti.opengl,
    }
    if target.backend not in arch_by_backend:
        raise ValueError(f"unsupported compiler backend: {target.backend}")

    output_root = Path(
        os.environ.get("PIXEL_REFINE_AOT_TCM_ROOT", str(DEFAULT_ARTIFACT_ROOT))
    ).expanduser()
    output = output_root / target.target_id / target.artifact_name("exposure_analyzer")
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing AOT artifact: {output}. "
            "Move it aside explicitly before recompiling."
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    arch = arch_by_backend[target.backend]
    ti.init(arch=arch, offline_cache=False)
    try:
        module = ti.aot.Module(arch)
        samples = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "samples", ti.f32, ndim=1
        )
        histogram = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "histogram", ti.i32, ndim=1
        )
        sample_count = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR, "sample_count", ti.i32
        )
        graph = ti.graph.GraphBuilder()
        graph.dispatch(_exposure_histogram_f32, samples, histogram, sample_count)
        module.add_graph("imgp_exposure_histogram_f32", graph.compile())
        module.archive(str(output))
    finally:
        ti.reset()

    print(f"[OK] Exposure Analyzer archived: {output}")
    print(
        f"[ABI] samples=f32[{EXPOSURE_SAMPLE_CAP}], "
        f"histogram=i32[{EXPOSURE_HISTOGRAM_BINS}], sample_count=i32"
    )
    print(f"[MEMORY] input={EXPOSURE_SAMPLE_CAP * 4} bytes; histogram={EXPOSURE_HISTOGRAM_BINS * 4} bytes")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan", "opengl"), required=True)
    parser.add_argument("--vendor", help="GPU vendor/name used in target identity")
    args = parser.parse_args()
    compile_exposure_analyzer(args.backend, args.vendor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
