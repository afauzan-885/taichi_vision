from setuptools import find_packages, setup


setup(
    name="taichi_vision",
    version="1.0.0",
    packages=find_packages(),
    # Package data is declared explicitly below.  Do not let stale
    # egg-info/SOURCES.txt files pull generated Nuitka extensions into a
    # release wheel and shadow the authoritative Python engine.
    include_package_data=False,
    package_data={
        # Target-qualified TCM archives are runtime inputs, not build cache.
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
