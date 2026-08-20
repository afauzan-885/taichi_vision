"""High-level backend decision manager.

This layer is intentionally side-effect free: compilation and runtime probes
can call it repeatedly while the public algorithm API remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import tempfile
from .capabilities import backend_candidates, classify_device
from taichi_vision.backend_config import normalize_backend


@dataclass
class BackendDecision:
    selected: str | None
    candidates: list[str]
    device: str
    reason: str
    # Keep the historical four positional fields intact while exposing the
    # actual filtered decision set.  Execution must consume ``eligible`` rather
    # than reinterpreting the original candidate list independently.
    eligible: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)


class BackendManager:
    def __init__(self, device="unknown", validated=None):
        self.device = device
        self.validated = (
            dict(validated) if validated is not None else self._load_runtime_status()
        )

    @staticmethod
    def _load_runtime_status():
        root = os.environ.get(
            "AOT_CACHE", os.path.join(tempfile.gettempdir(), "pixel_refine_aot_cache")
        )
        result = {}
        # Keep the status cache aligned with the canonical backend registry.
        # Omitting CUDA/GLES here made a persisted qualification invisible to
        # the decision layer even though their target-qualified artifacts were
        # present and the public selector could request them explicitly.
        for backend in ("cpu", "cuda", "vulkan", "opengl", "gles"):
            try:
                with open(
                    os.path.join(root, f"runtime_{backend}.json"), encoding="utf-8"
                ) as f:
                    result[backend] = json.load(f).get("status", "unknown")
            except (OSError, ValueError, TypeError):
                pass
        return result

    def _candidate_rejection(self, backend):
        """Return a bounded rejection reason or ``None`` when backend is eligible."""
        status = self.validated.get(backend, "unknown")
        caps = classify_device(self.device, backend)
        exact_intel_manifest = (
            backend == "vulkan"
            and caps.vendor == "intel"
            and caps.safe
            and "manifest validated" in caps.reason.lower()
        )
        # The legacy runtime cache is not keyed by device fingerprint, driver,
        # or artifact digest. An exact current Intel manifest is stronger
        # evidence and must supersede a stale generic quarantine.
        if status in ("quarantined", "unsupported") and not exact_intel_manifest:
            return caps.reason or status
        if not caps.safe:
            return caps.reason or status or "capability policy rejected backend"
        return None

    def decide(self, requested="auto"):
        requested = normalize_backend(requested, allow_auto=True)
        candidates = (
            [requested] if requested != "auto" else list(backend_candidates(self.device))
        )
        eligible = []
        rejected = {}
        for backend in candidates:
            reason = self._candidate_rejection(backend)
            if reason is not None:
                rejected[backend] = str(reason)
            else:
                eligible.append(backend)

        selected = eligible[0] if eligible else None
        if selected is not None:
            reason = "selected after capability/status filtering"
        else:
            details = "; ".join(
                f"{backend}: {rejected.get(backend, 'rejected')}"
                for backend in candidates
            )
            prefix = (
                f"explicit backend {requested!r} rejected"
                if requested != "auto"
                else "all automatic candidates rejected"
            )
            reason = prefix + (f": {details}" if details else "")

        return BackendDecision(
            selected,
            candidates,
            self.device,
            reason,
            eligible=eligible,
            rejected=rejected,
        )

    def run_with_fallback(self, operation, requested="auto"):
        """Run an operation against capability/status-approved backend contexts.

        ``operation(backend)`` must create/upload resources for that backend;
        this prevents accidental reuse of native buffers across contexts.
        Returns ``(result, backend, errors)``.
        """
        requested = normalize_backend(requested, allow_auto=True)
        decision = self.decide(requested)
        errors = {}
        strict = requested != "auto"

        if not decision.eligible:
            mode = "explicit backend" if strict else "automatic backend candidates"
            raise RuntimeError(
                f"{mode} rejected before execution: {decision.reason}"
            )

        # ``decide`` is the single authority for capability/status filtering.
        # Never iterate the unfiltered candidate list again here; doing so can
        # re-admit a backend that was rejected solely by capability policy.
        for backend in decision.eligible:
            try:
                return operation(backend), backend, errors
            except Exception as exc:
                errors[backend] = f"{type(exc).__name__}: {exc}"
                if strict:
                    break

        mode = "explicit backend" if strict else "automatic backend candidates"
        raise RuntimeError(f"{mode} failed without implicit fallback: {errors}")


def preflight_backend(device="unknown", requested="auto", validated=None):
    """Choose one backend before native buffers and graph resources exist.

    This is intentionally side-effect free.  A wrapper should use the
    returned decision to construct its engine and buffers as one context;
    fallback must never switch contexts midway through an active graph.
    """
    return BackendManager(device=device, validated=validated).decide(requested)
