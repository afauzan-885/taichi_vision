"""Memory-bounded exposure analysis for HDR reference-frame selection.

The host samples at most 65,536 RGB pixels and computes their linear luminance.
A fixed-size f32 input and 256-bin i32 histogram are then dispatched through
Taichi AOT. No full-resolution RGB image is uploaded.
"""

from dataclasses import dataclass

import numpy as np
import taichi as ti


EXPOSURE_HISTOGRAM_BINS = 256
EXPOSURE_SAMPLE_CAP = 65_536
_EXPOSURE_CENTER = 0.5
_EXPOSURE_SIGMA = 0.22


@dataclass(frozen=True)
class ExposureAnalysis:
    """Compact image exposure summary used to compare burst frames."""

    score: float
    median_luminance: float
    exposure_suitability: float
    usable_fraction: float
    sample_count: int


@ti.kernel
def _exposure_histogram_f32(
    samples: ti.types.ndarray(dtype=ti.f32, ndim=1),
    histogram: ti.types.ndarray(dtype=ti.i32, ndim=1),
    sample_count: ti.i32,
):
    for index in range(sample_count):
        value = ti.min(1.0, ti.max(0.0, samples[index]))
        bin_index = ti.min(
            EXPOSURE_HISTOGRAM_BINS - 1,
            ti.cast(value * EXPOSURE_HISTOGRAM_BINS, ti.i32),
        )
        ti.atomic_add(histogram[bin_index], 1)


def _sample_luminance(rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """Compute luminance for a deterministic, spatially uniform RGB sample."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB [H, W, 3], got {image.shape}")
    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError("RGB input must be floating-point linear RGB in [0, 1]")
    pixel_count = int(image.shape[0]) * int(image.shape[1])
    if pixel_count <= 0:
        raise ValueError("cannot analyze an empty image")

    stride = max(1, (pixel_count + EXPOSURE_SAMPLE_CAP - 1) // EXPOSURE_SAMPLE_CAP)
    sampled = image.reshape((-1, 3))[::stride]
    sample_count = min(int(sampled.shape[0]), EXPOSURE_SAMPLE_CAP)

    # The HDR frame itself stays in host RAM and is not copied into another
    # full-resolution RGB or luminance buffer.
    sample = np.asarray(sampled[:sample_count], dtype=np.float32)
    if not np.isfinite(sample).all():
        raise ValueError("sampled RGB pixels contain NaN or infinity")
    if np.any(sample < 0.0) or np.any(sample > 1.0):
        raise ValueError("sampled linear RGB values must be in [0, 1]")
    luminance = (
        sample[:, 0] * 0.299
        + sample[:, 1] * 0.587
        + sample[:, 2] * 0.114
    )
    np.nan_to_num(luminance, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    return np.ascontiguousarray(luminance, dtype=np.float32), sample_count


def _taichi_histogram(luminance: np.ndarray, sample_count: int) -> np.ndarray:
    """Run the compiled image-analysis graph; missing artifacts fail clearly."""
    from taichi_vision.taichi_algorithm.aot_api.research import _dispatch

    samples = np.zeros((EXPOSURE_SAMPLE_CAP,), dtype=np.float32)
    samples[:sample_count] = luminance[:sample_count]
    histogram = _dispatch(
        "exposure_analyzer",
        "imgp_exposure_histogram_f32",
        inputs={"samples": samples},
        outputs={
            "histogram": np.zeros((EXPOSURE_HISTOGRAM_BINS,), dtype=np.int32)
        },
        scalars={"sample_count": int(sample_count)},
    )
    return np.asarray(histogram, dtype=np.int64)


def analyze_exposure(rgb: np.ndarray) -> ExposureAnalysis:
    """Analyze one linear RGB frame for exposure suitability.

    Args:
        rgb: Floating-point ``(height, width, 3)`` RGB array in linear
            ``[0, 1]``. The call needs no backend or tuning arguments; it uses
            the active Taichi Vision target and its compiled artifact.

    Returns:
        Exposure score, median luminance, component scores, and sample count.

    This analyzer uses a target-qualified Taichi AOT graph. There is no
    execution fallback: if the active target artifact is missing, dispatch
    raises the loader's actionable ``FileNotFoundError``.
    """
    luminance, sample_count = _sample_luminance(rgb)
    counts = _taichi_histogram(luminance, sample_count)
    total = max(1, int(counts.sum()))
    probabilities = counts.astype(np.float64) / total
    centers = (np.arange(EXPOSURE_HISTOGRAM_BINS, dtype=np.float64) + 0.5) / (
        EXPOSURE_HISTOGRAM_BINS
    )
    exposure_suitability = float(
        np.sum(
            probabilities
            * np.exp(-0.5 * ((centers - _EXPOSURE_CENTER) / _EXPOSURE_SIGMA) ** 2)
        )
    )
    usable_fraction = float(probabilities[3:-3].sum())
    score = float(np.clip(0.7 * exposure_suitability + 0.3 * usable_fraction, 0, 1))
    median_index = int(np.searchsorted(np.cumsum(counts), total * 0.5, side="left"))
    median = float(centers[min(median_index, EXPOSURE_HISTOGRAM_BINS - 1)])
    return ExposureAnalysis(
        score=score,
        median_luminance=median,
        exposure_suitability=exposure_suitability,
        usable_fraction=usable_fraction,
        sample_count=sample_count,
    )


__all__ = [
    "EXPOSURE_HISTOGRAM_BINS",
    "EXPOSURE_SAMPLE_CAP",
    "ExposureAnalysis",
    "analyze_exposure",
]
