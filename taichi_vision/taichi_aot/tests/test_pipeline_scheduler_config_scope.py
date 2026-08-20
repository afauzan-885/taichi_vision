"""Regression tests for temporary block-pipeline configuration ownership."""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib.util
from pathlib import Path
import sys
import threading
import types

import pytest


_PIPELINE_PATH = Path(__file__).resolve().parents[1] / "pipeline_scheduler.py"
_MODULE_NAME = "taichi_aot_pipeline_scheduler_test_probe"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _PIPELINE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PIPELINE = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _PIPELINE
_SPEC.loader.exec_module(_PIPELINE)
PipelineStage = _PIPELINE.PipelineStage
run_block_pipeline = _PIPELINE.run_block_pipeline


@dataclass(frozen=True)
class _Config:
    enabled: bool = False
    size: int = 256
    threshold_bytes: int = 4096


class _FakeEngine:
    def __init__(self):
        self._config = _Config()
        self._config_lock = threading.Lock()

    def get_block_config(self):
        with self._config_lock:
            return replace(self._config)

    def configure_blocks(self, **kwargs):
        with self._config_lock:
            self._config = _Config(
                enabled=bool(kwargs.get("enabled", self._config.enabled)),
                size=int(kwargs.get("size", self._config.size)),
                threshold_bytes=int(
                    kwargs.get("threshold_bytes", self._config.threshold_bytes)
                ),
            )
            return self._config


class _FakeAOT(types.ModuleType):
    def __init__(self, engine):
        super().__init__("taichi_vision.taichi_aot")
        self.engine = engine

    @staticmethod
    def get_memory_status():
        return {
            "recommended_block_size": 256,
            "target_chunk_bytes": 4096,
        }

    def set_block_mode(self, enabled, *, size, threshold_bytes):
        return self.engine.configure_blocks(
            enabled=enabled,
            size=size,
            threshold_bytes=threshold_bytes,
        )


def _install_fake_aot(monkeypatch, engine):
    fake = _FakeAOT(engine)
    monkeypatch.setitem(sys.modules, "taichi_vision.taichi_aot", fake)
    return fake


def test_concurrent_pipelines_do_not_overwrite_active_config(monkeypatch):
    engine = _FakeEngine()
    _install_fake_aot(monkeypatch, engine)

    a_entered = threading.Event()
    b_call_started = threading.Event()
    allow_a_finish = threading.Event()
    b_stage_started = threading.Event()
    failures = []

    def stage_a(value):
        try:
            assert engine.get_block_config().size == 64
            a_entered.set()
            assert b_call_started.wait(1.0)
            # B has called run_block_pipeline, but it must still be waiting for
            # A's complete temporary-config scope rather than mutating the
            # shared engine configuration underneath A.
            assert not b_stage_started.wait(0.1)
            assert engine.get_block_config().size == 64
            assert allow_a_finish.wait(1.0)
            return value + "A"
        except BaseException as exc:  # propagate worker assertion to main thread
            failures.append(exc)
            allow_a_finish.set()
            return value

    def stage_b(value):
        b_stage_started.set()
        assert engine.get_block_config().size == 128
        return value + "B"

    def run_a():
        try:
            run_block_pipeline(
                "",
                [PipelineStage("A", stage_a)],
                block_size=64,
                threshold_bytes=100,
            )
        except BaseException as exc:
            failures.append(exc)
            allow_a_finish.set()

    def run_b():
        try:
            assert a_entered.wait(1.0)
            b_call_started.set()
            run_block_pipeline(
                "",
                [PipelineStage("B", stage_b)],
                block_size=128,
                threshold_bytes=200,
            )
        except BaseException as exc:
            failures.append(exc)
            allow_a_finish.set()

    thread_a = threading.Thread(target=run_a)
    thread_b = threading.Thread(target=run_b)
    thread_a.start()
    assert a_entered.wait(1.0)
    thread_b.start()
    assert b_call_started.wait(1.0)

    allow_a_finish.set()
    thread_a.join(timeout=2.0)
    thread_b.join(timeout=2.0)

    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert failures == []
    assert b_stage_started.is_set()
    assert engine.get_block_config() == _Config()


def test_exception_restores_original_engine_config(monkeypatch):
    engine = _FakeEngine()
    _install_fake_aot(monkeypatch, engine)

    def fail(_value):
        assert engine.get_block_config().size == 64
        raise RuntimeError("sentinel stage failure")

    with pytest.raises(RuntimeError, match="sentinel stage failure"):
        run_block_pipeline(
            object(),
            [PipelineStage("fail", fail)],
            block_size=64,
            threshold_bytes=123,
        )

    assert engine.get_block_config() == _Config()


def test_nested_same_thread_pipeline_restores_outer_then_original(monkeypatch):
    engine = _FakeEngine()
    _install_fake_aot(monkeypatch, engine)
    observed = []

    def inner(value):
        observed.append(("inner", engine.get_block_config().size))
        return value

    def outer(value):
        observed.append(("outer-before", engine.get_block_config().size))
        result = run_block_pipeline(
            value,
            [PipelineStage("inner", inner)],
            block_size=128,
            threshold_bytes=200,
        )
        observed.append(("outer-after", engine.get_block_config().size))
        return result

    run_block_pipeline(
        "value",
        [PipelineStage("outer", outer)],
        block_size=64,
        threshold_bytes=100,
    )

    assert observed == [
        ("outer-before", 64),
        ("inner", 128),
        ("outer-after", 64),
    ]
    assert engine.get_block_config() == _Config()
