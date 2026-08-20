"""Regression tests for bounded TCM validation and streaming payload checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from taichi_vision.taichi_aot import tcm_contract


def _write(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def test_validate_tcm_rejects_excessive_compression_ratio(tmp_path: Path) -> None:
    path = tmp_path / "ratio.tcm"
    _write(path, {"graphs.tcb": b"A" * (16 * 1024 * 1024)})

    with pytest.raises(tcm_contract.TcmContractError, match="compression ratio"):
        tcm_contract.validate_tcm(path)


def test_validate_tcm_rejects_member_budget_before_reading_payload(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "member-limit.tcm"
    _write(path, {"graphs.tcb": b"xx"})
    monkeypatch.setattr(tcm_contract, "_MAX_MEMBER_UNCOMPRESSED_BYTES", 1)

    with pytest.raises(tcm_contract.TcmContractError, match="too large"):
        tcm_contract.validate_tcm(path)


def test_validate_tcm_bounds_legacy_llvm_inspection(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "legacy.tcm"
    _write(path, {"graphs.tcb": b"graph", "kernel.ll": b"x" * 32})
    monkeypatch.setattr(tcm_contract, "_MAX_LEGACY_INSPECTION_BYTES", 8)

    with pytest.raises(tcm_contract.TcmContractError, match="LLVM inspection"):
        tcm_contract.validate_tcm(
            path,
            requested_target={
                "backend": "cpu",
                "arch": "x86_64",
                "os": "windows",
                "vendor": "unknown",
            },
        )


def test_validate_tcm_streams_manifest_payload_checksum(tmp_path: Path) -> None:
    path = tmp_path / "manifest.tcm"
    payload = b"native-payload" * 1024
    manifest = {
        "magic": "PIXEL_REFINE_TCM",
        "schema_version": 1,
        "tcm_format_version": 1,
        "compiler_version": "test",
        "minimum_runtime_abi": 1,
        "target": {
            "backend": "cpu",
            "arch": "x86_64",
            "os": "windows",
            "vendor": "unknown",
        },
        "payloads": [
            {
                "path": "kernel.bin",
                "kind": "native",
                "version": "test",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
        "kernels": [{"name": "main", "args": []}],
    }
    _write(
        path,
        {
            "kernel.bin": payload,
            "tcm_manifest.json": json.dumps(manifest).encode("utf-8"),
        },
    )

    report = tcm_contract.validate_tcm(path)
    assert report["status"] == "valid"
