"""Regression tests for bounded CPU AOT archive materialization."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import zipfile

import pytest

_engine_spec = importlib.util.spec_from_file_location(
    "taichi_vision.taichi_aot.engine_archive_test",
    Path(__file__).parents[1] / "engine.py",
)
assert _engine_spec and _engine_spec.loader
engine = importlib.util.module_from_spec(_engine_spec)
sys.modules[_engine_spec.name] = engine
_engine_spec.loader.exec_module(engine)


def _write_archive(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members:
            archive.writestr(name, payload)


def test_cpu_aot_materialization_extracts_valid_archive(tmp_path: Path) -> None:
    artifact = tmp_path / "valid.tcm"
    _write_archive(artifact, [("__content__", b""), ("graphs/demo.tcb", b"graph")])

    materialized = Path(engine._materialize_cpu_aot_directory(artifact))

    assert (materialized / "__content__").is_file()
    assert (materialized / "graphs" / "demo.tcb").read_bytes() == b"graph"


@pytest.mark.parametrize(
    ("limit", "limit_name", "members"),
    [
        (1, "_MAX_CPU_AOT_MEMBERS", [("__content__", b""), ("payload", b"x")]),
        (1, "_MAX_CPU_AOT_MEMBER_BYTES", [("__content__", b""), ("payload", b"xx")]),
        (1, "_MAX_CPU_AOT_TOTAL_BYTES", [("__content__", b""), ("payload", b"xx")]),
    ],
)
def test_cpu_aot_materialization_enforces_extraction_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int, limit_name: str, members
) -> None:
    artifact = tmp_path / "bounded.tcm"
    _write_archive(artifact, members)
    monkeypatch.setattr(engine, limit_name, limit)

    with pytest.raises(RuntimeError, match="CPU AOT"):
        engine._materialize_cpu_aot_directory(artifact)


def test_cpu_aot_materialization_rejects_duplicate_members(tmp_path: Path) -> None:
    artifact = tmp_path / "duplicate.tcm"
    _write_archive(artifact, [("__content__", b""), ("payload", b"a"), ("payload", b"b")])

    with pytest.raises(RuntimeError, match="Duplicate member"):
        engine._materialize_cpu_aot_directory(artifact)


def test_cpu_aot_materialization_rejects_path_traversal(tmp_path: Path) -> None:
    artifact = tmp_path / "traversal.tcm"
    _write_archive(artifact, [("__content__", b""), ("../escape", b"x")])

    with pytest.raises(RuntimeError, match="Unsafe member"):
        engine._materialize_cpu_aot_directory(artifact)
