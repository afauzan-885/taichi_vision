"""Regression tests for lazy adapter registration boundaries."""

from __future__ import annotations

import taichi_vision.taichi_aot.block_adapters as block_adapters
from taichi_vision.taichi_aot.compute_block import (
    ComputeBlockMetadata,
    _resolve_adapter_selection,
)


def test_unknown_decorator_name_does_not_register_all_default_adapters(monkeypatch):
    calls = []

    def forbidden(name):
        calls.append(name)
        raise AssertionError("unknown operation triggered global adapter registration")

    monkeypatch.setattr(block_adapters, "ensure_default_block_adapters", forbidden)
    canonical, adapter = _resolve_adapter_selection(
        ComputeBlockMetadata(name="my_module.my_helper")
    )

    # The source-only probe may expose either the canonical name or the
    # compatibility ``None`` result when a precompiled package cache is used;
    # both paths must remain free of global registration side effects.
    assert canonical in (None, "my_module.my_helper")
    assert adapter is None
    assert calls == []
