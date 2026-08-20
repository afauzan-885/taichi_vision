"""Reproducible CPU benchmark for canonical demosaic families."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import gc
import json
import os
import time
from pathlib import Path

import numpy as np


def _synthetic_bayer(height: int, width: int) -> np.ndarray:
    y = np.arange(height, dtype=np.float32)[:, None] / max(height - 1, 1)
    x = np.arange(width, dtype=np.float32)[None, :] / max(width - 1, 1)
    scene = np.empty((height, width, 3), dtype=np.float32)
    scene[..., 0] = 0.12 + 0.68 * x + 0.08 * y
    scene[..., 1] = 0.08 + 0.58 * y + 0.10 * x
    scene[..., 2] = 0.10 + 0.50 * (1.0 - x) + 0.08 * y
    scene += (0.025 * np.sin(x * 173.0) * np.cos(y * 137.0))[..., None]
    scene = np.clip(scene, 0.0, 1.0)
    bayer = np.empty((height, width), dtype=np.float32)
    bayer[0::2, 0::2] = scene[0::2, 0::2, 0]
    bayer[0::2, 1::2] = scene[0::2, 1::2, 1]
    bayer[1::2, 0::2] = scene[1::2, 0::2, 1]
    bayer[1::2, 1::2] = scene[1::2, 1::2, 2]
    return np.ascontiguousarray(bayer)


def _rss_mb() -> float | None:
    if os.name == "nt":
        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        try:
            counters = _Counters(); counters.cb = ctypes.sizeof(_Counters)
            api = ctypes.windll.psapi.GetProcessMemoryInfo
            api.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
            api.restype = wintypes.BOOL
            if api(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / (1024.0 * 1024.0)
        except Exception:
            pass
    return None


def run(method: str, height: int, width: int, warmup: int, repetitions: int) -> dict:
    os.environ.setdefault("AOT_ARCH", "cpu")
    os.environ.setdefault("AOT_BACKEND", "cpu")
    from taichi_vision.taichi_algorithm.aot_api import demosaic
    bayer = _synthetic_bayer(height, width)
    kwargs = dict(wb_r=1.0, wb_g1=1.0, wb_b=1.0, wb_g2=1.0,
                  cmatrix=np.eye(3, dtype=np.float32), black_level=0.0,
                  white_level=1.0, c00=0, c01=1, c10=1, c11=2,
                  method=method, return_gpu=False)
    for _ in range(max(0, warmup)):
        output = demosaic(bayer, **kwargs)
        if not isinstance(output, np.ndarray) or output.shape != (height, width, 3) or not np.isfinite(output).all():
            raise RuntimeError(f"invalid {method} warmup output")
    gc.collect(); rss_before = _rss_mb(); samples = []; output = None
    for _ in range(max(1, repetitions)):
        started = time.perf_counter(); output = demosaic(bayer, **kwargs)
        samples.append(time.perf_counter() - started)
        if not isinstance(output, np.ndarray) or output.shape != (height, width, 3) or not np.isfinite(output).all():
            raise RuntimeError(f"invalid {method} output")
    samples.sort(); median = float(np.median(samples)); p95 = float(np.percentile(samples, 95)); pixels = height * width
    return {"method": method, "width": width, "height": height, "megapixels": pixels / 1e6,
            "warmup": warmup, "repetitions": repetitions, "samples_s": samples,
            "median_s": median, "p95_s": p95, "median_mp_per_s": pixels / median / 1e6,
            "p95_mp_per_s": pixels / p95 / 1e6, "rss_before_mb": rss_before,
            "rss_after_mb": _rss_mb(), "dtype": str(output.dtype),
            "output_min": float(output.min()), "output_max": float(output.max()),
            "output_mean": float(output.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("bilinear", "hamilton", "arm", "dcb"))
    parser.add_argument("--width", type=int, default=4000); parser.add_argument("--height", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=1); parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path); args = parser.parse_args()
    payload = json.dumps(run(args.method, args.height, args.width, args.warmup, args.repetitions), sort_keys=True)
    print(payload, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
