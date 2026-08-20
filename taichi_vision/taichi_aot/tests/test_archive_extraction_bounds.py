"""Regression tests for bounded CPU AOT archive materialization."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import textwrap
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


def _run_materialization_processes(
    artifact: Path, project_root: Path, engine_source: Path, roles: list[str], tmp_path: Path
) -> list[tuple[int, str, str]]:
    script = textwrap.dedent(
        f"""
        import importlib.util
        import os
        import sys
        import time

        spec = importlib.util.spec_from_file_location(
            'taichi_vision.taichi_aot.engine_source', {str(engine_source)!r}
        )
        engine = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = engine
        spec.loader.exec_module(engine)
        role = os.environ.get('AOT_TEST_ROLE', 'normal')
        if role == 'delay':
            rename = engine.os.rename
            def delayed_rename(source, target):
                time.sleep(0.5)
                return rename(source, target)
            engine.os.rename = delayed_rename
        if role == 'fail':
            archive_open = engine.zipfile.ZipFile.open
            def failing_open(archive, member, *args, **kwargs):
                if getattr(member, 'filename', '') == 'graphs/demo.tcb':
                    raise RuntimeError('injected staging failure')
                return archive_open(archive, member, *args, **kwargs)
            engine.zipfile.ZipFile.open = failing_open
        if role == 'crash':
            archive_open = engine.zipfile.ZipFile.open
            def crashing_open(archive, member, *args, **kwargs):
                if getattr(member, 'filename', '') == 'graphs/demo.tcb':
                    os._exit(23)
                return archive_open(archive, member, *args, **kwargs)
            engine.zipfile.ZipFile.open = crashing_open
        if role == 'reuse':
            def reject_extraction(**kwargs):
                raise RuntimeError('unexpected extraction')
            engine.tempfile.mkdtemp = reject_extraction
        try:
            print('PATH:' + engine._materialize_cpu_aot_directory({str(artifact)!r}), flush=True)
        except Exception as error:
            if role == 'fail':
                print('EXPECTED_FAILURE:' + str(error), flush=True)
                sys.exit(0)
            raise
        """
    )
    base_env = os.environ.copy()
    base_env.update(
        {
            "AOT_ARCH": "cpu",
            "PIXEL_REFINE_AOT_ARCH": "cpu",
            "PIXEL_REFINE_BACKEND": "cpu",
            "PYTHONPATH": str(project_root),
            "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache"),
        }
    )
    workers = []
    for role in roles:
        env = base_env.copy()
        env["AOT_TEST_ROLE"] = role
        workers.append(
            subprocess.Popen(
                [sys.executable, "-c", script],
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    results = []
    for worker in workers:
        stdout, stderr = worker.communicate(timeout=60)
        output_lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        results.append((worker.returncode, output_lines[-1] if output_lines else "", stderr))
    return results


def test_cpu_aot_materialization_is_safe_across_processes(tmp_path: Path) -> None:
    artifact = tmp_path / "shared.tcm"
    _write_archive(
        artifact,
        [("__content__", b""), ("graphs/demo.tcb", b"graph" * 4096)],
    )
    project_root = Path(__file__).parents[3]
    engine_source = Path(__file__).parents[1] / "engine.py"
    results = _run_materialization_processes(
        artifact, project_root, engine_source, ["delay", "normal", "normal", "normal"], tmp_path
    )

    assert all(code == 0 for code, _, _ in results), results
    paths = {stdout.removeprefix("PATH:") for _, stdout, _ in results}
    assert len(paths) == 1
    materialized = Path(next(iter(paths)))
    assert (materialized / "__content__").is_file()
    assert (materialized / "graphs" / "demo.tcb").read_bytes() == b"graph" * 4096


def test_cpu_aot_materialization_cleans_failed_staging_and_recovers(tmp_path: Path) -> None:
    artifact = tmp_path / "failure.tcm"
    _write_archive(artifact, [("__content__", b""), ("graphs/demo.tcb", b"graph")])
    project_root = Path(__file__).parents[3]
    engine_source = Path(__file__).parents[1] / "engine.py"

    failed = _run_materialization_processes(artifact, project_root, engine_source, ["fail"], tmp_path)
    assert failed[0][0] == 0, failed
    assert failed[0][1].startswith("EXPECTED_FAILURE:")
    cache_root = artifact.parent / ".cpu_aot_cache"
    assert not list(cache_root.glob("extract-*"))

    recovered = _run_materialization_processes(
        artifact, project_root, engine_source, ["normal", "normal"], tmp_path
    )
    assert all(code == 0 for code, _, _ in recovered), recovered


def test_cpu_aot_materialization_recovers_after_process_crash(tmp_path: Path) -> None:
    artifact = tmp_path / "crash.tcm"
    _write_archive(artifact, [("__content__", b""), ("graphs/demo.tcb", b"graph")])
    project_root = Path(__file__).parents[3]
    engine_source = Path(__file__).parents[1] / "engine.py"

    crashed = _run_materialization_processes(artifact, project_root, engine_source, ["crash"], tmp_path)
    assert crashed[0][0] == 23, crashed
    recovered = _run_materialization_processes(artifact, project_root, engine_source, ["normal"], tmp_path)
    assert recovered[0][0] == 0, recovered
    assert recovered[0][1].startswith("PATH:")


def test_cpu_aot_materialization_reuses_preexisting_target_without_extraction(tmp_path: Path) -> None:
    artifact = tmp_path / "reuse.tcm"
    _write_archive(artifact, [("__content__", b""), ("graphs/demo.tcb", b"graph")])
    project_root = Path(__file__).parents[3]
    engine_source = Path(__file__).parents[1] / "engine.py"

    created = _run_materialization_processes(artifact, project_root, engine_source, ["normal"], tmp_path)
    assert created[0][0] == 0, created
    reused = _run_materialization_processes(artifact, project_root, engine_source, ["reuse"], tmp_path)
    assert reused[0][0] == 0, reused
    assert reused[0][1].startswith("PATH:")
