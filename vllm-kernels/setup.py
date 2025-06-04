# SPDX-License-Identifier: Apache-2.0
import importlib.util
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import List, Optional, Tuple

import torch
from packaging.version import Version, parse
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

ROOT_DIR = Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)

# # --- Environment Variable Handling (Simplified for vllm-kernels) ---
# VLLM_TARGET_DEVICE will determine which kernels to build (cuda, rocm, cpu)
# For vllm-kernels, we assume this is primarily for GPU.
# It can be set externally, e.g., VLLM_TARGET_DEVICE=cuda pip install .
VLLM_TARGET_DEVICE = os.getenv("VLLM_TARGET_DEVICE")

if VLLM_TARGET_DEVICE is None:
    if torch.cuda.is_available() and CUDA_HOME:
        VLLM_TARGET_DEVICE = "cuda"
    elif torch.version.hip and ROCM_HOME: # Placeholder for ROCm check
        VLLM_TARGET_DEVICE = "rocm"
    else:
        VLLM_TARGET_DEVICE = "cpu" # Fallback, though most kernels are GPU-specific
    logger.info(f"VLLM_TARGET_DEVICE not set, determined: {VLLM_TARGET_DEVICE}")
else:
    logger.info(f"VLLM_TARGET_DEVICE set to: {VLLM_TARGET_DEVICE}")

# --- Helper Functions (Adapted from root setup.py) ---
def _is_cuda() -> bool:
    return VLLM_TARGET_DEVICE == "cuda" and CUDA_HOME is not None

def _is_rocm() -> bool: # Renamed from _is_hip for clarity in this context
    return VLLM_TARGET_DEVICE == "rocm" and ROCM_HOME is not None and torch.version.hip is not None

def _is_cpu() -> bool:
    return VLLM_TARGET_DEVICE == "cpu"

def get_nvcc_cuda_version() -> Optional[Version]:
    if not _is_cuda() or CUDA_HOME is None:
        return None
    nvcc_path = Path(CUDA_HOME) / "bin" / "nvcc"
    if not nvcc_path.exists():
        logger.warning("NVCC not found at %s", nvcc_path)
        return None
    try:
        nvcc_output = subprocess.check_output([str(nvcc_path), "-V"], universal_newlines=True)
        match = re.search(r"release (\d+\.\d+)", nvcc_output)
        if match:
            return parse(match.group(1))
    except Exception as e:
        logger.warning(f"Failed to get CUDA version from nvcc: {e}")
    return None

def is_ninja_available() -> bool:
    return which("ninja") is not None

# --- CMake Build Classes (Adapted from root setup.py) ---
class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = '.', **kwargs) -> None:
        super().__init__(name, sources=[], **kwargs)
        self.cmake_lists_dir = str(Path(cmake_lists_dir).resolve())


class CMakeBuild(build_ext):
    _did_config: dict[str, bool] = {}

    def run(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("CMake must be installed to build vllm-kernels") from e

        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_extension(ext)
        super().run() # For any other non-CMake extensions if added later

    def _get_build_args(self, ext: CMakeExtension) -> List[str]:
        # Path where the .so files should be placed to be importable by the package
        # For vllm_kernels._C, output_dir would be build_lib/vllm_kernels
        # For vllm_kernels.flash_attn._vllm_fa2_C, it would be build_lib/vllm_kernels/flash_attn
        output_dir = Path(self.build_lib) / ext.name.replace(".", os.sep).rpartition(os.sep)[0]
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DVLLM_TARGET_DEVICE={VLLM_TARGET_DEVICE}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        
        build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release" if not self.debug else "Debug")
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")

        if _is_cuda() and CUDA_HOME:
            cmake_args.append(f"-DCMAKE_CUDA_COMPILER={Path(CUDA_HOME) / 'bin' / 'nvcc'}")
        
        if which("sccache"):
            cmake_args.extend([
                "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache",
            ])
        elif which("ccache"):
             cmake_args.extend([
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
            ])

        verbose = os.environ.get("VERBOSE", "0") == "1"
        if verbose:
            cmake_args.append("-DCMAKE_VERBOSE_MAKEFILE=ON")
            
        return cmake_args

    def build_extension(self, ext: CMakeExtension) -> None:
        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        ext_dir.mkdir(parents=True, exist_ok=True)

        build_temp = Path(self.build_temp) / ext.name.split('.')[-1] # Use short name for build temp dir
        if not build_temp.exists():
            build_temp.mkdir(parents=True)
            
        cmake_lists_dir = Path(ext.cmake_lists_dir)

        # Check if CMake configuration for this directory has already been done
        # This simple check might need refinement if multiple extensions share a CMakeLists.txt
        # but have different configurations (not the case here as all use ROOT_DIR).
        if str(cmake_lists_dir) not in self._did_config:
            cmake_args = self._get_build_args(ext)
            
            build_tool_args = []
            if is_ninja_available():
                 build_tool_args = ["-G", "Ninja"]
            
            logger.info(f"Configuring CMake for extensions in {cmake_lists_dir} with args: {build_tool_args + cmake_args}")
            subprocess.check_call(["cmake", str(cmake_lists_dir)] + build_tool_args + cmake_args, cwd=str(build_temp))
            self._did_config[str(cmake_lists_dir)] = True
        else:
            logger.info(f"CMake already configured for {cmake_lists_dir}")

        # Build specific target for the extension
        # The target name in CMakeLists.txt is the short name (e.g., _C)
        target_name = ext.name.split('.')[-1]
        num_jobs = os.cpu_count()
        max_jobs = os.getenv("MAX_JOBS")
        if max_jobs:
            num_jobs = int(max_jobs)

        build_args = ["--build", ".", "--target", target_name, f"-j{num_jobs}"]
        logger.info(f"Building CMake target {target_name} with args: {build_args}")
        subprocess.check_call(["cmake"] + build_args, cwd=str(build_temp))


# --- Define Kernel Extensions ---
# The names here must match the module names expected by the Python binding code.
# e.g., "vllm_kernels._C" will produce _C.so inside the vllm_kernels package.
ext_modules = []

if _is_cuda() or _is_rocm() or _is_cpu(): # _C is generic but often CUDA-focused
    ext_modules.append(CMakeExtension(name="vllm_kernels._C", cmake_lists_dir=str(ROOT_DIR)))

if _is_cuda() or _is_rocm(): # MoE ops are typically GPU-specific
    ext_modules.append(CMakeExtension(name="vllm_kernels._moe_C", cmake_lists_dir=str(ROOT_DIR)))

if _is_rocm():
    ext_modules.append(CMakeExtension(name="vllm_kernels._rocm_C", cmake_lists_dir=str(ROOT_DIR))) # Specific ROCm ops

if _is_cuda():
    ext_modules.append(CMakeExtension(name="vllm_kernels.flash_attn._vllm_fa2_C", cmake_lists_dir=str(ROOT_DIR)))
    
    nvcc_version = get_nvcc_cuda_version()
    if nvcc_version and nvcc_version >= Version("12.3"): # FA3 requires CUDA 12.3+
        ext_modules.append(CMakeExtension(name="vllm_kernels.flash_attn._vllm_fa3_C", cmake_lists_dir=str(ROOT_DIR)))
    
    # FlashMLA might be optional or specific to certain architectures
    ext_modules.append(CMakeExtension(name="vllm_kernels._flashmla_C", cmake_lists_dir=str(ROOT_DIR), optional=True))
    
    ext_modules.append(CMakeExtension(name="vllm_kernels.cumem_allocator", cmake_lists_dir=str(ROOT_DIR)))


setup(
    name="vllm-kernels",
    # Version will be managed by pyproject.toml or CI
    # version="0.0.1", # Placeholder, ideally read from pyproject.toml or git
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    # Other metadata like author, description, license will be in pyproject.toml
    packages=['vllm_kernels', 'vllm_kernels.flash_attn'],
) 