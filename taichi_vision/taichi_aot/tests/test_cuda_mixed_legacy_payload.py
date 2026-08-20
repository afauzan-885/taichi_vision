"""Parity test for mixed host-helper/NVPTX legacy CUDA archives."""

from __future__ import annotations

from pathlib import Path
import zipfile

from taichi_vision.taichi_aot.artifact_targets import TargetSpec, resolve_artifact
from taichi_vision.taichi_aot.tcm_preflight import preflight_tcm


def test_mixed_host_and_nvptx_payload_passes_both_validators(tmp_path: Path) -> None:
    target = TargetSpec(backend="cuda", arch="x86_64", os="windows", vendor="nvidia")
    path = tmp_path / target.target_id / target.artifact_name("mixed")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("__version__", "1.7.4")
        archive.writestr("graphs.tcb", b"graph")
        archive.writestr("host.ll", b'target triple = "x86_64-pc-windows-msvc19.44.35228"\n')
        archive.writestr("device.ll", b'target triple = "nvptx64-nvidia-cuda"\n')

    assert resolve_artifact(tmp_path, "mixed", target, allow_legacy=False) == path
    decision = preflight_tcm(path, requested_target=target.as_dict())
    assert decision.allowed is True
    assert decision.status == "legacy"
