# SPDX-License-Identifier: Apache-2.0
import importlib.util
import logging
import os
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools_scm import get_version as scm_get_version

# Utility to load a module from a path
def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

ROOT_DIR = Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)

# Load envs.py from vllm-engine/vllm/envs.py
# This is used by get_vllm_version for VLLM_TARGET_DEVICE, etc.
try:
    envs = load_module_from_path('envs', ROOT_DIR / 'vllm' / 'envs.py')
except FileNotFoundError:
    # Fallback if envs.py is not critical or found, or define defaults
    class MockEnvs:
        VLLM_TARGET_DEVICE = os.getenv("VLLM_TARGET_DEVICE", "cuda") # Default or detect
        VLLM_USE_PRECOMPILED = os.getenv("VLLM_USE_PRECOMPILED")
        # Add other envs used by get_vllm_version if any, with defaults
    envs = MockEnvs()
    logger.info("vllm/envs.py not found or not fully loaded for setup.py, using defaults/env variables.")


# Versioning
# Adapted from the original setup.py's get_vllm_version
# This function is simplified as it doesn't need to know about CUDA/ROCM versions for the engine package itself,
# but relies on setuptools_scm. The device-specific versioning is for vllm-kernels.
def get_engine_version() -> str:
    try:
        # Version will be like '0.4.0.post1+git.12345' or '0.4.1'
        version = scm_get_version(
            root="..", # Assuming setup.py is in vllm-engine, so root is one level up for git versioning from top level.
                      # If .git is in vllm-engine, then root="."
            relative_to=__file__,
            write_to=ROOT_DIR / "vllm" / "_version.py",
            # Example: version_scheme="release-branch-semver", local_scheme="node-and-date"
        )
    except Exception as e:
        logger.warning(f"setuptools_scm failed to get version: {e}. Falling back to a default or error.")
        # Fallback or re-raise, depending on policy. For now, let it error out if scm fails.
        # Ensure there's a pyproject.toml with [tool.setuptools_scm] if needed.
        # For now, assume .git is at the workspace root.
        # If .git is not available in the CI/build environment, this needs a fallback.
        version = scm_get_version(write_to=ROOT_DIR / "vllm" / "_version.py") # Try with default root
    return version


# Requirements
# Simplified get_requirements. Reads from vllm-engine/requirements/common.txt
# and adds vllm-kernels.
def get_engine_requirements() -> list[str]:
    requirements_dir = ROOT_DIR / "requirements"
    common_req_file = requirements_dir / "common.txt"
    
    resolved_requirements = []
    if common_req_file.exists():
        with open(common_req_file) as f:
            requirements = f.read().strip().split("\\n")
        for line in requirements:
            if line.startswith("-r "):
                # In case common.txt itself references other files, though we simplify for now
                # For simplicity, assume direct dependencies in common.txt
                pass # Potentially recurse or handle -r if used in engine's common.txt
            elif not line.startswith("--") and not line.startswith("#") and line.strip() != "":
                resolved_requirements.append(line)
    else:
        logger.warning(f"{common_req_file} not found. Engine dependencies might be incomplete.")

    # Add vllm-kernels as a core dependency
    # Potentially add version specifier, e.g., f"vllm-kernels=={get_engine_version()}"
    # if versions are kept in sync and vllm-kernels is already versioned by a similar mechanism.
    # For now, a loose dependency.
    resolved_requirements.append("vllm-kernels")
    
    # Ensure torch is listed if not in common.txt, as vllm-engine directly uses it.
    # The vllm-engine/pyproject.toml already lists torch in its [build-system].requires,
    # but this is for runtime.
    # Example: if "torch" not in " ".join(resolved_requirements):
    #    resolved_requirements.append("torch>=2.7.0") # Match version from original pyproject

    return resolved_requirements

setup(
    # Name, author, license, etc., are primarily defined in pyproject.toml [project] table.
    # setup.py is now mainly for dynamic parts like version and dependencies.
    version=get_engine_version(),
    packages=find_packages(where=".", include=['vllm', 'vllm.*']),
    install_requires=get_engine_requirements(),
    # python_requires can also be in pyproject.toml. Ensure consistency.
    # Example: python_requires=">=3.9,<3.13",
    
    # Entry points are defined in pyproject.toml [project.scripts]
    # and should be picked up automatically by setuptools if this setup.py is used.
) 