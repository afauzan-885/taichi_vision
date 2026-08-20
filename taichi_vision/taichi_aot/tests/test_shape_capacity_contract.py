"""Regression tests for checked tensor shape and byte-capacity arithmetic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))
import taichi_vision  # noqa: F401,E402

_SPEC = importlib.util.spec_from_file_location(
    "taichi_vision.taichi_aot.shape_capacity_engine",
    Path(__file__).parents[1] / "engine.py",
)
assert _SPEC and _SPEC.loader
engine = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = engine
_SPEC.loader.exec_module(engine)


def test_checked_shape_capacity_uses_python_integer_arithmetic() -> None:
    shape, nbytes = engine._checked_shape_nbytes((2, 3, 4), np.float32)

    assert shape == (2, 3, 4)
    assert nbytes == 96


@pytest.mark.parametrize(
    "shape",
    [
        (True, 2),
        (0, 2),
        (-1, 2),
        (1.5, 2),
        (2**31, 2),
        tuple(range(1, 10)),
    ],
)
def test_checked_shape_capacity_rejects_invalid_or_narrowed_shapes(shape) -> None:
    with pytest.raises(ValueError):
        engine._checked_shape_nbytes(shape, np.float32)


def test_checked_shape_capacity_rejects_uint64_byte_overflow() -> None:
    dimension = 2**31 - 1

    with pytest.raises(OverflowError, match="uint64"):
        engine._checked_shape_nbytes((dimension, dimension, dimension, 2), np.float64)
