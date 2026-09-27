"""Native packed-tile accumulation and in-place normalization.

The input tile already contains weighted RGB sums and scalar coverage.
This API deliberately applies no second window or coverage weighting.
"""
import os
from pathlib import Path

_module_cache = {}


def _module(engine):
    from taichi_vision.taichi_aot.artifact_targets import detect_target, resolve_artifact
    root = (os.environ.get("PIXEL_REFINE_RESIDENT_ACCUMULATOR_TCM_ROOT")
            or os.environ.get("PIXEL_REFINE_AOT_TCM_ROOT")
            or str(Path(__file__).parent / "aot_tcm"))
    live = engine._live() if hasattr(engine, "_live") else engine
    key = (id(live), live._generation, os.path.abspath(root))
    if key in _module_cache:
        return _module_cache[key]
    target = detect_target(backend=engine.arch, device=engine.gpu_name)
    path = resolve_artifact(root, "resident_accumulator", target, allow_legacy=False)
    if path is None:
        raise FileNotFoundError(f"Resident accumulator TCM missing for {target.target_id}")
    module = engine.load(str(path))
    _module_cache.clear()
    _module_cache[key] = module
    return module


def _validate(numerator, denominator, y, x, h, w):
    import numpy as np
    if numerator.dtype != np.float32 or denominator.dtype != np.float32:
        raise ValueError("accumulator buffers must have float32 dtype")
    if len(numerator.shape) != 3 or numerator.shape[2] != 3 or denominator.shape != numerator.shape[:2]:
        raise ValueError("accumulator requires matching (H,W,3) and (H,W) buffers")
    if min(y, x) < 0 or min(h, w) < 1 or y+h > denominator.shape[0] or x+w > denominator.shape[1]:
        raise ValueError("tile lies outside the native accumulator")


def accumulate_packed_tile(packed, numerator, denominator, *, offset_y=0, offset_x=0):
    from taichi_vision.taichi_aot.engine import engine
    if packed.dtype != numerator.dtype or packed.dtype != denominator.dtype:
        raise ValueError("accumulator buffers must share float32 dtype")
    if len(packed.shape) != 3 or packed.shape[2] != 4:
        raise ValueError("packed tile must have shape (H,W,4)")
    _validate(numerator, denominator, offset_y, offset_x, *packed.shape[:2])
    _module(engine).run(
        "cmn_accumulate_packed_rgb_tile", packed=packed, numerator=numerator,
        denominator=denominator, offset_y=int(offset_y), offset_x=int(offset_x),
    )


def normalize_accumulator_tile(numerator, denominator, fallback, *, offset_y=0,
                               offset_x=0, height, width, fallback_y=0, fallback_x=0):
    from taichi_vision.taichi_aot.engine import engine
    _validate(numerator, denominator, offset_y, offset_x, height, width)
    if (len(fallback.shape) != 3 or fallback.shape[2] != 3 or fallback.dtype != numerator.dtype
            or min(fallback_y, fallback_x) < 0 or fallback_y+height > fallback.shape[0]
            or fallback_x+width > fallback.shape[1]):
        raise ValueError("fallback tile does not cover the normalized region")
    _module(engine).run(
        "cmn_normalize_rgb_tile_inplace", numerator=numerator, denominator=denominator,
        fallback=fallback, offset_y=int(offset_y), offset_x=int(offset_x),
        height=int(height), width=int(width), fallback_y=int(fallback_y), fallback_x=int(fallback_x),
    )



if os.environ.get("AOT_MODE") == "0":
    import taichi as ti

    @ti.kernel
    def _accumulate(packed: ti.types.ndarray(dtype=ti.f32, ndim=3),
                    numerator: ti.types.ndarray(dtype=ti.f32, ndim=3),
                    denominator: ti.types.ndarray(dtype=ti.f32, ndim=2),
                    offset_y: ti.i32, offset_x: ti.i32):
        for y, x in ti.ndrange(packed.shape[0], packed.shape[1]):
            for c in ti.static(range(3)):
                numerator[y + offset_y, x + offset_x, c] += packed[y, x, c]
            denominator[y + offset_y, x + offset_x] += packed[y, x, 3]

    @ti.kernel
    def _normalize(numerator: ti.types.ndarray(dtype=ti.f32, ndim=3),
                   denominator: ti.types.ndarray(dtype=ti.f32, ndim=2),
                   fallback: ti.types.ndarray(dtype=ti.f32, ndim=3),
                   offset_y: ti.i32, offset_x: ti.i32, height: ti.i32, width: ti.i32,
                   fallback_y: ti.i32, fallback_x: ti.i32):
        for y, x in ti.ndrange(height, width):
            gy, gx = y + offset_y, x + offset_x
            weight = denominator[gy, gx]
            for c in ti.static(range(3)):
                value = fallback[y + fallback_y, x + fallback_x, c]
                if weight > 1e-8:
                    value = numerator[gy, gx, c] / weight
                numerator[gy, gx, c] = value

    def build_graphs(module):
        def array(name, ndim):
            return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype=ti.f32, ndim=ndim)
        def scalar(name):
            return ti.graph.Arg(ti.graph.ArgKind.SCALAR, name, dtype=ti.i32)
        num, den = array("numerator", 3), array("denominator", 2)
        oy, ox = scalar("offset_y"), scalar("offset_x")
        graph = ti.graph.GraphBuilder()
        graph.dispatch(_accumulate, array("packed", 3), num, den, oy, ox)
        module.add_graph("cmn_accumulate_packed_rgb_tile", graph.compile())
        graph = ti.graph.GraphBuilder()
        graph.dispatch(_normalize, num, den, array("fallback", 3), oy, ox,
                       scalar("height"), scalar("width"), scalar("fallback_y"), scalar("fallback_x"))
        module.add_graph("cmn_normalize_rgb_tile_inplace", graph.compile())
