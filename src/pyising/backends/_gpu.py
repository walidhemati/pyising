"""GPU environment helpers shared by the GPU backends."""

from __future__ import annotations

import ctypes
import functools
import os
import warnings


@functools.cache
def prepare_cupy() -> None:
    """Make CuPy's NVRTC loadable from the pip ``nvidia-cuda-nvrtc`` wheel when it is not on the
    loader path (pip-only CUDA installs), and silence the duplicate-package warning."""
    warnings.filterwarnings("ignore", category=UserWarning, module="cupy._environment")
    try:
        import nvidia.cuda_nvrtc as nvrtc  # type: ignore[import-not-found]

        for base in nvrtc.__path__:
            lib = os.path.join(base, "lib", "libnvrtc.so.12")
            if os.path.exists(lib):
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                break
    except (ImportError, OSError):
        pass


@functools.cache
def prepare_numba() -> None:
    """numba-cuda 0.30 still references ``np.row_stack`` (removed in NumPy 2.5)."""
    import numpy as np

    if not hasattr(np, "row_stack"):
        np.row_stack = np.vstack  # type: ignore[attr-defined]
    warnings.filterwarnings("ignore", category=Warning, module="numba")
    os.environ.setdefault("NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS", "0")


def cuda_available() -> bool:
    """True when torch sees a CUDA device."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
