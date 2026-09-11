# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

import ctypes
import glob
import logging
from pathlib import Path
from typing import Final

logger: Final = logging.getLogger(__name__)


def setup_gpu_environment() -> None:
    """
    Preload CUDA/cuDNN libraries installed inside the Python environment.

    TensorFlow may not find pip/uv-provided NVIDIA libraries through the
    system linker, so load them before importing TensorFlow.

    An alternative is to configure the LD_LIBRARY_PATH.
    """
    try:
        import site

        site_dirs = site.getsitepackages()
        if hasattr(site, "getusersitepackages"):
            site_dirs.append(site.getusersitepackages())

        loaded_count = 0
        for site_pkg in site_dirs:
            nvidia_path = Path(site_pkg) / "nvidia"
            if nvidia_path.is_dir():
                for lib_dir in nvidia_path.glob("*/lib"):
                    for so_file in sorted(glob.glob(f"{lib_dir}/*.so*")):
                        try:
                            ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
                            loaded_count += 1
                        except Exception:
                            pass

        if loaded_count > 0:
            logger.debug("Pre-loaded %d NVIDIA CUDA/cuDNN shared libraries.", loaded_count)
    except Exception as e:
        logger.debug("GPU library pre-loader encountered: %s", e)
