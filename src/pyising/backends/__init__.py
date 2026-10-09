"""Backend registry. Every entry is ``(backend, variant) -> "module:Class"``; modules are imported
lazily so a missing package or a missing GPU only disables that entry."""

from __future__ import annotations

import importlib
from functools import cache

from .base import IsingSimulation

REGISTRY: dict[tuple[str, str], str] = {
    ("numpy", "int8"): "numpy_backend:NumpyCheckerboard",
    ("numpy", "msc"): "numpy_backend:NumpyMultiSpin",
    ("torch-eager", "int8"): "torch_backend:TorchEager",
    ("torch-compile", "int8"): "torch_backend:TorchCompile",
    ("triton", "int8"): "triton_backend:TritonInt8",
    ("triton", "msc"): "triton_backend:TritonMultiSpin",
    ("warp", "int8"): "warp_backend:WarpInt8",
    ("warp", "msc"): "warp_backend:WarpMultiSpin",
    ("numba", "int8"): "numba_backend:NumbaInt8",
    ("numba", "msc"): "numba_backend:NumbaMultiSpin",
    ("cupy-cuda-c", "int8"): "cupy_backend:CupyInt8",
    ("cupy-cuda-c", "int8-rng-per-site"): "cupy_backend:CupyInt8RngPerSite",
    ("cupy-cuda-c", "msc"): "cupy_backend:CupyMultiSpin",
    ("cupy-cuda-c", "msc-eager32"): "cupy_backend:CupyMultiSpinEager",
}

#: The order of the optimisation ladder (also the plotting order).
BACKENDS = ("numpy", "torch-eager", "torch-compile", "triton", "warp", "numba", "cupy-cuda-c")
GPU_ONLY = {"triton", "warp", "numba", "cupy-cuda-c"}


@cache
def load(backend: str, variant: str = "int8") -> type[IsingSimulation] | None:
    """Return the simulation class, or ``None`` if it cannot be imported / used here."""
    key = (backend, variant)
    if key not in REGISTRY:
        raise KeyError(f"unknown backend/variant {backend}/{variant}; known: {sorted(REGISTRY)}")
    if backend in GPU_ONLY or backend.startswith("torch"):
        from ._gpu import cuda_available

        if backend in GPU_ONLY and not cuda_available():
            return None
    mod_name, cls_name = REGISTRY[key].split(":")
    try:
        mod = importlib.import_module(f"{__name__}.{mod_name}")
    except Exception:  # noqa: BLE001 - any import failure just disables the backend
        return None
    return getattr(mod, cls_name, None)


def available() -> list[tuple[str, str]]:
    """All registry entries that can run on this machine."""
    return [k for k in REGISTRY if load(*k) is not None]


def create(backend: str, variant: str = "int8", **kwargs) -> IsingSimulation:
    """Instantiate a simulation; raises ``RuntimeError`` if the backend is unavailable."""
    cls = load(backend, variant)
    if cls is None:
        raise RuntimeError(f"backend {backend}/{variant} is not available on this machine")
    return cls(**kwargs)


__all__ = ["REGISTRY", "BACKENDS", "IsingSimulation", "load", "available", "create"]
