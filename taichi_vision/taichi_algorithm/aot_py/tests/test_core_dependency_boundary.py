"""Guard the production algorithm boundary from reference backends."""

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CORE_FILES = (
    PROJECT_ROOT / "taichi_vision" / "taichi_algorithm" / "aot_api" / "__init__.py",
    PROJECT_ROOT / "taichi_vision" / "taichi_algorithm" / "alignment" / "mtb.py",
    PROJECT_ROOT / "taichi_vision" / "taichi_algorithm" / "alignment" / "ransac.py",
    PROJECT_ROOT / "taichi_vision" / "taichi_algorithm" / "feature_matching" / "ofb.py",
    PROJECT_ROOT / "taichi_vision" / "taichi_algorithm" / "optical_flow" / "lucas_kanade.py",
    PROJECT_ROOT / "taichi_vision" / "taichi_algorithm" / "sfm" / "five_point_solver.py",
)
IMPORT_RE = re.compile(r"^\s*(?:from\s+cv2\s+import|import\s+cv2)\b", re.MULTILINE)


def test_core_algorithm_modules_do_not_import_opencv():
    for path in CORE_FILES:
        source = path.read_text(encoding="utf-8")
        assert not IMPORT_RE.search(source), f"OpenCV import leaked into core: {path}"


def test_aot_facade_has_no_opencv_call_site():
    source = CORE_FILES[0].read_text(encoding="utf-8")
    assert "cv2." not in source
