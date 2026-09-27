"""Resident SR preparation with exact float32 exposure order statistics."""
import os
from pathlib import Path
import numpy as np

_cache = {}


def _native_f32(buffer):
    if getattr(buffer, "handle", None) is None or buffer.dtype != np.float32:
        raise ValueError("resident preparation requires a live native float32 buffer")


def _module():
    from taichi_vision.taichi_aot.engine import engine
    from taichi_vision.taichi_aot.artifact_targets import detect_target, resolve_artifact
    root = os.environ.get("PIXEL_REFINE_SR_RESIDENT_TCM_ROOT") or os.environ.get("PIXEL_REFINE_AOT_TCM_ROOT") or str(Path(__file__).parent / "aot_tcm")
    live = engine._live()
    key = (id(live), live._generation, os.path.abspath(root))
    if key not in _cache:
        target = detect_target(backend=engine.arch, device=engine.gpu_name)
        path = resolve_artifact(root, "sr_resident", target, allow_legacy=False)
        if path is None:
            raise FileNotFoundError(f"SR preparation TCM missing for {target.target_id}")
        _cache.clear()
        _cache[key] = engine.load(str(path))
    return _cache[key]


def _view(buffer):
    return buffer.view_as_vector(False) if buffer.is_vector else buffer


def resident_exposure_gain(reference, support, *, session):
    _native_f32(reference)
    _native_f32(support)
    if reference.shape != support.shape or len(reference.shape) != 2:
        raise ValueError("exposure requires matching native grayscale planes")
    chunks = (reference.shape[0] * reference.shape[1] + 2047) // 2048
    hist = session.acquire_buffer((chunks, 2, 256), dtype=np.int32, tag="sr.exposure.hist")
    totals = session.acquire_buffer((2, 256), dtype=np.int32, tag="sr.exposure.totals")
    state = session.acquire_buffer((2, 3), dtype=np.int32, tag="sr.exposure.state")
    gain = session.acquire_buffer((1,), dtype=np.float32, tag="sr.exposure.gain")
    module = _module()
    module.run("cmn_sr_exposure_init", state=state)
    for shift in (24, 16, 8, 0):
        module.run("cmn_sr_exposure_radix", reference=reference, support=support,
                   hist=hist, totals=totals, state=state, shift=shift)
    module.run("cmn_sr_exposure_gain", state=state, gain=gain)
    return gain


def resident_scale(src, *, session, gain=None, factor=1.0, dst=None):
    _native_f32(src)
    if dst is None:
        from taichi_vision import taichi_aot
        dst = session.own(taichi_aot.engine.allocate(src.shape, dtype=np.float32))
    if gain is None:
        gain = session.acquire_buffer((1,), dtype=np.float32, tag="sr.unit_gain")
        _module().run("cmn_sr_unit_gain", gain=gain)
    ndim = len(src.shape)
    _native_f32(dst)
    _native_f32(gain)
    if gain.shape != (1,):
        raise ValueError("resident gain must be a one-element buffer")
    if ndim not in (2, 3) or dst.shape != src.shape:
        raise ValueError("resident scale requires matching 2-D or 3-D buffers")
    _module().run(f"cmn_sr_scale_{ndim}d", src=_view(src), dst=_view(dst), gain=gain, factor=float(factor))
    return dst


def resident_warp_gray(src, flow, *, session):
    from taichi_vision import taichi_aot
    _native_f32(src)
    _native_f32(flow)
    if len(src.shape) != 2 or flow.shape != (*src.shape, 2):
        raise ValueError("resident warp requires matching grayscale and dense flow")
    return taichi_aot.remap_with_flow(src, flow, *src.shape, return_gpu=True, session=session)


def resident_crop_rgb(src, y, x, height, width, *, session):
    from taichi_vision import taichi_aot
    _native_f32(src)
    if (len(src.shape) != 3 or src.shape[2] != 3 or min(y, x) < 0
            or min(height, width) < 1 or y+height > src.shape[0] or x+width > src.shape[1]):
        raise ValueError("resident RGB crop lies outside its source")
    dst = session.own(taichi_aot.engine.allocate((height, width, 3), dtype=np.float32))
    _module().run("cmn_sr_crop_rgb", src=_view(src), dst=_view(dst), y=int(y), x=int(x))
    return dst


def resident_resize_flow(src, height, width, *, factor, session):
    from taichi_vision import taichi_aot
    _native_f32(src)
    if len(src.shape) != 3 or src.shape[2] != 2 or min(height, width) < 1:
        raise ValueError("resident flow resize requires (H,W,2) and positive output dimensions")
    dst = session.own(taichi_aot.engine.allocate((height, width, 2), dtype=np.float32))
    _module().run("cmn_sr_resize_flow", src=_view(src), dst=_view(dst), factor=float(factor))
    return dst


if os.environ.get("AOT_MODE") == "0":
    import taichi as ti

    @ti.kernel
    def _init(state: ti.types.ndarray(ti.i32, ndim=2)):
        for i, j in state:
            state[i, j] = 0

    @ti.kernel
    def _clear(hist: ti.types.ndarray(ti.i32, ndim=3)):
        for i, j, k in hist:
            hist[i, j, k] = 0

    @ti.kernel
    def _hist(ref: ti.types.ndarray(ti.f32, ndim=2), supp: ti.types.ndarray(ti.f32, ndim=2),
              hist: ti.types.ndarray(ti.i32, ndim=3), state: ti.types.ndarray(ti.i32, ndim=2), shift: ti.i32):
        for y, x in ref:
            a, b = ref[y, x], supp[y, x]
            if a > .03 and b > .03 and a < .98 and b < .98:
                bits = ti.bit_cast(a / ti.max(b, 1e-4), ti.i32)
                for rank in ti.static(range(2)):
                    matches = shift == 24
                    if shift < 24:
                        matches = (bits >> (shift + 8)) == (state[rank, 0] >> (shift + 8))
                    if matches:
                        ti.atomic_add(hist[(y * ref.shape[1] + x) // 2048, rank, (bits >> shift) & 255], 1)

    @ti.kernel
    def _sum(hist: ti.types.ndarray(ti.i32, ndim=3), totals: ti.types.ndarray(ti.i32, ndim=2)):
        for rank, bucket in totals:
            count = 0
            for chunk in range(hist.shape[0]):
                count += hist[chunk, rank, bucket]
            totals[rank, bucket] = count

    @ti.kernel
    def _choose(totals: ti.types.ndarray(ti.i32, ndim=2), state: ti.types.ndarray(ti.i32, ndim=2), shift: ti.i32):
        for rank in range(2):
            if shift == 24:
                count = 0
                for bucket in range(256):
                    count += totals[rank, bucket]
                state[rank, 2] = count
                state[rank, 1] = ti.max(0, (count - 1 + rank) // 2)
            remaining = state[rank, 1]
            chosen = -1
            for bucket in range(256):
                if chosen < 0:
                    count = totals[rank, bucket]
                    if remaining < count:
                        chosen = bucket
                    else:
                        remaining -= count
            if chosen >= 0:
                state[rank, 0] = state[rank, 0] | (chosen << shift)
                state[rank, 1] = remaining

    @ti.kernel
    def _gain(state: ti.types.ndarray(ti.i32, ndim=2), gain: ti.types.ndarray(ti.f32, ndim=1)):
        value = 1.0
        if state[0, 2] >= 32:
            value = ti.min(2.0, ti.max(.5, (ti.bit_cast(state[0, 0], ti.f32) + ti.bit_cast(state[1, 0], ti.f32)) * .5))
        if ti.abs(value - 1.0) <= 1e-5:
            value = 1.0
        gain[0] = value

    @ti.kernel
    def _unit(gain: ti.types.ndarray(ti.f32, ndim=1)):
        gain[0] = 1.0

    @ti.kernel
    def _scale2(src: ti.types.ndarray(ti.f32, ndim=2), dst: ti.types.ndarray(ti.f32, ndim=2), gain: ti.types.ndarray(ti.f32, ndim=1), factor: ti.f32):
        for y, x in src:
            dst[y, x] = src[y, x] * gain[0] * factor

    @ti.kernel
    def _scale3(src: ti.types.ndarray(ti.f32, ndim=3), dst: ti.types.ndarray(ti.f32, ndim=3), gain: ti.types.ndarray(ti.f32, ndim=1), factor: ti.f32):
        for y, x, c in src:
            dst[y, x, c] = src[y, x, c] * gain[0] * factor

    @ti.kernel
    def _crop(src: ti.types.ndarray(ti.f32, ndim=3), dst: ti.types.ndarray(ti.f32, ndim=3), y: ti.i32, x: ti.i32):
        for iy, ix, c in dst:
            dst[iy, ix, c] = src[iy+y, ix+x, c]

    @ti.kernel
    def _resize_flow(src: ti.types.ndarray(ti.f32, ndim=3), dst: ti.types.ndarray(ti.f32, ndim=3), factor: ti.f32):
        for y, x in ti.ndrange(dst.shape[0], dst.shape[1]):
            sx = ti.max(0.0, ti.min(ti.cast(src.shape[1]-1, ti.f32), (x+.5)*src.shape[1]/dst.shape[1]-.5))
            sy = ti.max(0.0, ti.min(ti.cast(src.shape[0]-1, ti.f32), (y+.5)*src.shape[0]/dst.shape[0]-.5))
            ix, iy = ti.cast(ti.floor(sx), ti.i32), ti.cast(ti.floor(sy), ti.i32)
            jx, jy = ti.min(ix+1, src.shape[1]-1), ti.min(iy+1, src.shape[0]-1)
            fx, fy = sx-ix, sy-iy
            for c in ti.static(range(2)):
                dst[y, x, c] = ((src[iy, ix, c]*(1-fx)+src[iy, jx, c]*fx)*(1-fy)
                                +(src[jy, ix, c]*(1-fx)+src[jy, jx, c]*fx)*fy)*factor

    def build_graphs(module):
        def a(name, ndim, dtype=ti.f32):
            return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype=dtype, ndim=ndim)
        def s(name, dtype=ti.i32):
            return ti.graph.Arg(ti.graph.ArgKind.SCALAR, name, dtype=dtype)
        def add(name, calls):
            graph = ti.graph.GraphBuilder()
            for kernel, args in calls:
                graph.dispatch(kernel, *args)
            module.add_graph(name, graph.compile())
        state, hist, totals = a("state", 2, ti.i32), a("hist", 3, ti.i32), a("totals", 2, ti.i32)
        shift, gain = s("shift"), a("gain", 1)
        add("cmn_sr_exposure_init", [(_init, [state])])
        add("cmn_sr_exposure_radix", [(_clear, [hist]), (_hist, [a("reference", 2), a("support", 2), hist, state, shift]), (_sum, [hist, totals]), (_choose, [totals, state, shift])])
        add("cmn_sr_exposure_gain", [(_gain, [state, gain])])
        add("cmn_sr_unit_gain", [(_unit, [gain])])
        for ndim, kernel in ((2, _scale2), (3, _scale3)):
            add(f"cmn_sr_scale_{ndim}d", [(kernel, [a("src", ndim), a("dst", ndim), gain, s("factor", ti.f32)])])
        add("cmn_sr_crop_rgb", [(_crop, [a("src", 3), a("dst", 3), s("y"), s("x")])])
        add("cmn_sr_resize_flow", [(_resize_flow, [a("src", 3), a("dst", 3), s("factor", ti.f32)])])
