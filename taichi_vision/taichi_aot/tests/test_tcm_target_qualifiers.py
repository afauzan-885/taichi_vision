"""Regression tests for ABI/variant-qualified TCM target matching."""

from __future__ import annotations

import pytest

from taichi_vision.taichi_aot.tcm_contract import TcmContractError, validate_manifest


def _manifest(**target_overrides):
    target = {
        "backend": "cpu",
        "arch": "x86_64",
        "os": "windows",
        "vendor": "unknown",
        **target_overrides,
    }
    return {
        "magic": "PIXEL_REFINE_TCM",
        "schema_version": 1,
        "tcm_format_version": 1,
        "compiler_version": "test",
        "minimum_runtime_abi": 1,
        "target": target,
        "payloads": [{"path": "kernel.bin", "kind": "native"}],
        "kernels": [{"name": "main", "args": []}],
    }


def test_matching_abi_and_variant_are_accepted():
    report = validate_manifest(
        _manifest(abi="msvc-static", variant="baseline"),
        requested_target={
            "backend": "cpu",
            "arch": "x86_64",
            "os": "windows",
            "abi": "MSVC-STATIC",
            "variant": "BASELINE",
        },
    )
    assert report["target"]["abi"] == "msvc-static"
    assert report["target"]["variant"] == "baseline"


@pytest.mark.parametrize("field", ["abi", "variant"])
def test_qualifier_mismatch_is_rejected(field):
    target = {field: "optimized"}
    requested = {
        "backend": "cpu",
        "arch": "x86_64",
        "os": "windows",
        field: "baseline",
    }
    with pytest.raises(TcmContractError, match=field):
        validate_manifest(_manifest(**target), requested_target=requested)
