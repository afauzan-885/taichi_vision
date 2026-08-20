from setuptools import setup, find_packages


_SUBPACKAGES = find_packages()
_PACKAGES = ["taichi_vision", *[f"taichi_vision.{name}" for name in _SUBPACKAGES]]

setup(
    name="taichi_vision",
    version="1.0.0",
    # setup.py lives inside the package directory in this repository.  Map
    # that directory explicitly so the wheel preserves the public
    # ``taichi_vision.*`` namespace instead of exposing implementation
    # packages such as ``taichi_algorithm`` at top level.
    packages=_PACKAGES,
    package_dir={"taichi_vision": "."},
    # Package data is declared explicitly below.  Do not let stale
    # egg-info/SOURCES.txt files pull generated Nuitka extensions into a
    # release wheel and shadow the authoritative Python engine.
    include_package_data=False,
    package_data={
        # Target-qualified algorithm graphs are runtime inputs, not build
        # cache output.  Keep every supported target in the wheel so backend
        # selection remains deterministic on another machine.
        "taichi_vision.taichi_algorithm": [
            "aot_tcm/*/*.tcm",
            "aot_tcm/*/*/*.tcm",
            "aot_tcm/target_manifest.json",
            "aot_py/aot_dll/*/*.dll",
            "aot_py/aot_dll/*/*.pyd",
            "aot_py/aot_dll/*/*.so",
            "aot_py/runtime_smoke_manifest.json",
            "aot_py/raw_icd_gl_dispatch.h",
        ],
        # Native bridges and the small manifest/header files used by the
        # loader/validation helpers must travel with the Python package.
        "taichi_vision.taichi_algorithm.aot_py": [
            "aot_dll/*/*.dll",
            "aot_dll/*/*.pyd",
            "aot_dll/*/*.so",
            "runtime_smoke_manifest.json",
            "raw_icd_gl_dispatch.h",
        ],
        # The Python engine is authoritative.  Do not package a stale
        # Nuitka engine*.pyd beside engine.py: Windows import precedence
        # would silently bypass the process-owned runtime hardening.
    },
    install_requires=[
        "taichi==1.7.4",
        "numpy>=2.0.0",
    ],
    description="Shared Taichi iGPU algorithm library for Pixel Refine",
    python_requires=">=3.12.9",
)
