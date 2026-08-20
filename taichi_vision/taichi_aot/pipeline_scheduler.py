"""Safe block-oriented composition for large image pipelines.

This scheduler intentionally composes the existing public algorithm callables
instead of recording one oversized graphics graph.  Each callable may use the
normal AOT block executor; intermediate host arrays are released as soon as
the next stage owns the result.  The API is internal and does not alter the
algorithm functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Any
import gc
import threading
import importlib


class PipelineCancelledError(RuntimeError):
    """Raised when a cooperative block-pipeline cancellation is requested."""

    def __init__(self, stage_index: int = 0):
        self.stage_index = int(stage_index)
        self.reason = "cancel_check"
        super().__init__(f"block pipeline cancelled before stage {self.stage_index}")

    def as_dict(self) -> dict[str, object]:
        """Return bounded, JSON-safe cancellation telemetry for UI/logging."""

        return {
            "cancelled": True,
            "stage_index": self.stage_index,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PipelineStage:
    name: str
    operation: Callable[[Any], Any]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("pipeline stage name must be a non-empty string")
        if not callable(self.operation):
            raise TypeError("pipeline stage operation must be callable")


_PIPELINE_SCOPE_LOCK_GUARD = threading.Lock()


def _pipeline_scope_lock(engine):
    """Return one re-entrant block-policy scope lock for a live engine.

    ``run_block_pipeline`` temporarily mutates the engine-wide block policy.
    A dedicated lock keeps those snapshot/set/restore scopes from interleaving
    without holding the engine's native-dispatch lock across arbitrary host
    stage code.  RLock preserves well-defined same-thread nesting.
    """

    lock = getattr(engine, "_block_pipeline_scope_lock", None)
    if lock is not None:
        return lock
    with _PIPELINE_SCOPE_LOCK_GUARD:
        lock = getattr(engine, "_block_pipeline_scope_lock", None)
        if lock is None:
            lock = threading.RLock()
            setattr(engine, "_block_pipeline_scope_lock", lock)
        return lock


def run_block_pipeline(source, stages: Iterable[PipelineStage], *,
                       block_size: int | None = None,
                       threshold_bytes: int | None = None,
                       cancel_check: Callable[[], bool] | None = None):
    """Run a dependency-ordered pipeline through safe block-capable APIs.

    The scheduler is deliberately host-array based: this avoids mixing native
    OpenGL graph recording with host fallbacks.  Existing operations decide
    their own halo and full-frame policy, while this function controls memory
    pressure and stage ordering.

    The block policy is engine-global, so the complete temporary override is a
    serialized per-engine scope.  This prevents another concurrent pipeline
    from overwriting the active policy or restoring a snapshot captured inside
    this pipeline's override.
    """
    if cancel_check is not None and not callable(cancel_check):
        raise TypeError("cancel_check must be callable or None")
    if cancel_check is not None and bool(cancel_check()):
        raise PipelineCancelledError(0)
    # Resolve through the import system so embedded callers/tests that provide
    # a process-local facade get the same module object as every other runtime
    # entry point.  Attribute-style package imports can retain a stale child
    # module after an intentional runtime replacement.
    aot = importlib.import_module("taichi_vision.taichi_aot")
    memory = aot.get_memory_status()
    if block_size is None:
        block_size = int(memory["recommended_block_size"])
    if threshold_bytes is None:
        threshold_bytes = max(1, int(memory["target_chunk_bytes"]))
    if block_size <= 0 or threshold_bytes < 0:
        raise ValueError("block_size must be positive and threshold_bytes non-negative")

    scope_lock = _pipeline_scope_lock(aot.engine)
    with scope_lock:
        previous = aot.engine.get_block_config()
        aot.set_block_mode(
            True,
            size=int(block_size),
            threshold_bytes=int(threshold_bytes),
        )
        value = source
        try:
            for stage_index, stage in enumerate(stages):
                if cancel_check is not None and bool(cancel_check()):
                    raise PipelineCancelledError(stage_index)
                if not isinstance(stage, PipelineStage):
                    raise TypeError("stages must contain PipelineStage values")
                next_value = stage.operation(value)
                if next_value is None:
                    raise RuntimeError(f"pipeline stage '{stage.name}' returned None")
                if next_value is not value:
                    del value
                    gc.collect()
                value = next_value
            return value
        finally:
            aot.engine.configure_blocks(**previous.__dict__)
