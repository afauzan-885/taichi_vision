"""Cross-layer DynamicArg validation contract tests."""

from __future__ import annotations

from pathlib import Path


CPP = (Path(__file__).parents[1] / "taichi_aot_engine.cpp").read_text(encoding="utf-8")
PY = (Path(__file__).parents[3] / "taichi_aot" / "engine.py").read_text(encoding="utf-8")


def test_native_descriptor_rejects_unknown_enums_and_bounds_before_memory_use():
    body = CPP[CPP.index("static bool _fill_ti_arg") : CPP.index("// -----------------------------------------------------------------------", CPP.index("static bool _fill_ti_arg") + 1)]
    assert "dyn_arg.arg_type != 0 && dyn_arg.arg_type != 1" in body
    assert "dyn_arg.dtype < 0 || dyn_arg.dtype > 5" in body
    assert "dyn_arg.dim_count < 1 || dyn_arg.dim_count > 8" in body
    assert "dyn_arg.elem_dim_count < 0 || dyn_arg.elem_dim_count > 8" in body
    assert "dyn_arg.shape[d] <= 0" in body


def test_python_descriptor_path_rejects_unrepresentable_dtypes_and_shapes():
    assert "_dtype_code_by_dtype[val_dtype]" in PY
    assert "unsupported GPU buffer dtype" in PY
    assert "buffer rank must be in 1.." in PY
    assert "positive INT32 values" in PY
    assert "expected_engine=engine" in PY
