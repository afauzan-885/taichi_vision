"""Cross-layer backend identity and explicit-selection contracts."""

from __future__ import annotations

from pathlib import Path


ENGINE = (Path(__file__).parents[1] / "engine.py").read_text(encoding="utf-8")
NATIVE = (
    Path(__file__).parents[2] / "taichi_algorithm" / "aot_py" / "taichi_aot_engine.cpp"
).read_text(encoding="utf-8")


def test_explicit_taichi_arch_never_substitutes_another_backend():
    assert "refusing to substitute a different backend" in ENGINE


def test_native_cpu_fallback_reports_actual_x64_architecture():
    fallback_start = NATIVE.index("PIXEL_REFINE_AOT_ALLOW_CPU_FALLBACK")
    fallback = NATIVE[fallback_start : NATIVE.index("return (void *)ctx", fallback_start)]
    assert "ctx->arch = TI_ARCH_X64" in fallback
    assert "new ti::Runtime(TI_ARCH_X64, 0)" in fallback
