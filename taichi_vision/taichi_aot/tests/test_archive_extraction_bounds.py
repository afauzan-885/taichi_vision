"""Regression tests for bounded CPU AOT archive materialization."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import zipfile

import pytest
sys.path.insert(0, str(Path(__file__).parents[3]))
import taichi_vision

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


def test_cpu_aot_materialization_is_safe_across_processes(tmp_path: Path) -> None:
    artifact = tmp_path / "shared.tcm"
    _write_archive(
        artifact,
        [("__content__", b""), ("graphs/demo.tcb", b"graph" * 4096)],
    )
    project_root = Path(__file__).parents[3]
    engine_source = Path(__file__).parents[1] / "engine.py"
    script = (
        "import importlib.util, sys; "
        f"spec = importlib.util.spec_from_file_location('taichi_vision.taichi_aot.engine_source', {str(engine_source)!r}); "
        "engine = importlib.util.module_from_spec(spec); "
        "sys.modules[spec.name] = engine; spec.loader.exec_module(engine); "
        f"print(engine._materialize_cpu_aot_directory({str(artifact)!r}), flush=True)"
    )
    env = os.environ.copy()
    env.update(
        {
            "AOT_ARCH": "cpu",
            "PIXEL_REFINE_AOT_ARCH": "cpu",
            "PIXEL_REFINE_BACKEND": "cpu",
            "PYTHONPATH": str(project_root),
            "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache"),
        }
    )
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    results = []
    for worker in workers:
        stdout, stderr = worker.communicate(timeout=60)
        output_lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        results.append((worker.returncode, output_lines[-1] if output_lines else "", stderr))

    assert all(code == 0 for code, _, _ in results), results
    paths = {stdout for _, stdout, _ in results}
    assert len(paths) == 1
    materialized = Path(next(iter(paths)))
    assert (materialized / "__content__").is_file()
    assert (materialized / "graphs" / "demo.tcb").read_bytes() == b"graph" * 4096
