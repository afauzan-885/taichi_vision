"""Focused WebP lossless production matrix.

This is a validation-only harness. NumPy creates deterministic fixtures and
Pillow is used only as an independent external decoder; neither is imported
by the native codec runtime. The matrix covers odd dimensions, 1/3/4 channels,
effort modes, metadata headers, deterministic output, and external decode parity.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from .webp_aot import encode_webp_lossless_aot, parse_webp_aot


def _decode_webp_external(payload: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as img:
        return np.array(img)


def _fixture(height: int, width: int, channels: int) -> np.ndarray:
    yy, xx = np.indices((int(height), int(width)), dtype=np.uint32)
    red = (xx * 17 + yy * 5 + (xx ^ yy)) & 255
    green = (xx * 3 + yy * 19 + ((xx * yy) & 63)) & 255
    blue = ((xx * 29) ^ (yy * 11) ^ (xx * yy * 3)) & 255
    if channels == 1:
        return ((red + green + blue) // 3).astype(np.uint8)
    if channels == 3:
        return np.stack((red, green, blue), axis=-1).astype(np.uint8)
    if channels == 4:
        alpha = ((xx * 7 + yy * 13) & 127) + 128
        return np.stack((red, green, blue, alpha), axis=-1).astype(np.uint8)
    raise ValueError(f"unsupported channel count: {channels}")


def _validate(encoded: bytes, source: np.ndarray, *, channels: int, effort: str) -> dict[str, Any]:
    if not encoded.startswith(b"RIFF") or b"WEBP" not in encoded[:16]:
        raise AssertionError("WebP RIFF container signature is missing")
    
    chunks = parse_webp_aot(encoded)
    chunk_kinds = [item[0] for item in chunks]
    if b"VP8L" not in chunk_kinds:
        raise AssertionError(f"expected VP8L lossless chunk in WebP stream, found {chunk_kinds}")
    
    decoded = _decode_webp_external(encoded)
    if channels == 1:
        expected_img = np.stack([source, source, source], axis=-1)
    else:
        expected_img = source
    
    if list(decoded.shape) != list(expected_img.shape):
        raise AssertionError(f"decoded shape mismatch: {decoded.shape} != {expected_img.shape}")
    
    # Check bit-exact reconstruction for lossless WebP
    if not np.array_equal(decoded, expected_img):
        max_diff = int(np.max(np.abs(decoded.astype(int) - expected_img.astype(int))))
        raise AssertionError(f"lossless WebP is not bit-exact, max_diff={max_diff}")

    return {
        "bytes": len(encoded),
        "decoded_shape": list(decoded.shape),
        "lossless_exact": True,
    }


def run_matrix() -> dict[str, Any]:
    dimensions = ((1, 1), (7, 9), (16, 17), (32, 32), (64, 48))
    channels_list = (1, 3, 4)
    efforts = ("fast", "baseline", "best")
    cases: list[dict[str, Any]] = []

    for height, width in dimensions:
        for channels in channels_list:
            image = _fixture(height, width, channels)
            for effort in efforts:
                encoded = encode_webp_lossless_aot(image, effort=effort)
                repeated = encode_webp_lossless_aot(image, effort=effort)
                if encoded != repeated:
                    raise AssertionError("WebP output is not deterministic")
                
                result = _validate(encoded, image, channels=channels, effort=effort)
                cases.append(
                    {
                        "shape": [height, width, channels],
                        "channels": channels,
                        "effort": effort,
                        "bytes": result["bytes"],
                        "lossless_exact": True,
                    }
                )

    # Metadata test
    metadata_test_image = _fixture(32, 32, 3)
    meta = {
        "EXIF": b"Exif\x00\x00II*\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    }
    meta_encoded = encode_webp_lossless_aot(metadata_test_image, effort="baseline", metadata=meta)
    meta_chunks = parse_webp_aot(meta_encoded)
    meta_kinds = [item[0] for item in meta_chunks]
    if b"EXIF" not in meta_kinds or b"VP8X" not in meta_kinds:
        raise AssertionError("WebP metadata chunk not recognized by parser")
    
    meta_decoded = _decode_webp_external(meta_encoded)
    if not np.array_equal(meta_decoded, metadata_test_image):
        raise AssertionError("Metadata WebP pixel data corrupted")

    return {
        "passed": True,
        "case_count": len(cases),
        "metadata_case_passed": True,
        "all_lossless_exact": True,
        "dimensions": [list(d) for d in dimensions],
        "channels": list(channels_list),
        "efforts": list(efforts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args()
    report = run_matrix()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
