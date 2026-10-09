"""CuPy ``RawModule`` backends: the hand-written CUDA C reference (compiled with NVRTC)."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from ._gpu import prepare_cupy

prepare_cupy()
import cupy as cp  # noqa: E402
import torch  # noqa: E402

from ..lattice import from_compressed, pack_bits, to_compressed, unpack_bits  # noqa: E402
from ..philox import split_seed  # noqa: E402
from .base import IsingSimulation  # noqa: E402
from .observables_torch import observables_int8, observables_packed  # noqa: E402

SOURCE = (Path(__file__).parent / "cuda" / "ising.cu").read_text()
THREADS = 256


@functools.cache
def _module(defines: tuple[tuple[str, int], ...]) -> cp.RawModule:
    opts = ("--std=c++17", "-use_fast_math") + tuple(f"-D{k}={v}" for k, v in defines)
    return cp.RawModule(code=SOURCE, options=opts)


_SPLITMIX = r"""
    unsigned long long z = (unsigned long long)i + seed * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
"""
_hot_int8 = cp.ElementwiseKernel("uint64 seed", "int8 s", _SPLITMIX + "s = (z >> 63) ? 1 : -1;",
                                 "pyising_hot_int8")
_hot_u64 = cp.ElementwiseKernel("uint64 seed", "uint64 w", _SPLITMIX + "w = z;", "pyising_hot_u64")


def _as_torch(a: cp.ndarray) -> torch.Tensor:
    return torch.from_dlpack(a)


class CupyInt8(IsingSimulation):
    """int8 compressed checkerboard; one thread per 4 sites (``char4``), one Philox per thread."""

    backend = "cupy-cuda-c"
    variant = "int8"
    rng_per_site = 0

    def _setup(self) -> None:
        self.ly = self.shape[-2]
        self.lz = self.shape[0] if self.ndim == 3 else 1
        defs = (("LH", self.lh), ("LY", self.ly), ("LZ", self.lz), ("NDIM", self.ndim),
                ("HEATBATH", int(self.algorithm == "heatbath")),
                ("RNG_PER_SITE", self.rng_per_site))
        self.kernel = _module(defs).get_function("int8_half")
        self.c = [cp.ones((self.rows, self.lh), dtype=cp.int8) for _ in range(2)]
        self.k0, self.k1 = split_seed(self.seed)
        self.n_quads = self.rows * self.lh // 4
        self._upload_table()

    def _upload_table(self) -> None:
        self.table_d = cp.asarray(self.table)

    def _half_sweep(self, color: int) -> None:
        blocks = (self.n_quads + THREADS - 1) // THREADS
        self.kernel((blocks,), (THREADS,),
                    (self.c[color], self.c[1 - color], self.table_d, np.int64(self.n_quads),
                     np.uint32(self.t), np.uint32(color), np.uint32(self.k0), np.uint32(self.k1)))

    def get_lattice(self) -> np.ndarray:
        return from_compressed(cp.asnumpy(self.c[0]), cp.asnumpy(self.c[1]), self.shape)

    def set_lattice(self, lat: np.ndarray) -> None:
        c0, c1 = to_compressed(np.asarray(lat, dtype=np.int8).reshape(self.shape))
        self.c = [cp.asarray(c0), cp.asarray(c1)]

    def _fill_hot(self) -> None:
        """i.i.d. spins from a splitmix64 hash of the index, in place (no temporaries)."""
        for c, arr in enumerate(self.c):
            _hot_int8(np.uint64(2 * self.seed + c), arr)

    def synchronize(self) -> None:
        cp.cuda.Device().synchronize()

    def observables(self) -> tuple[float, float]:
        return observables_int8(_as_torch(self.c[0]), _as_torch(self.c[1]), self.shape)

    def magnetization(self) -> float:
        return self.observables()[0]

    def energy(self) -> float:
        return self.observables()[1]


class CupyInt8RngPerSite(CupyInt8):
    """Ablation: identical numbers, but one Philox call per site instead of per 4 sites."""

    variant = "int8-rng-per-site"
    rng_per_site = 1


class CupyMultiSpin(IsingSimulation):
    """Multi-spin coding: 64 spins per uint64, bit-sliced neighbour count, lazy bitwise
    comparison of the uniforms (expected ~ log2(64) random bits per spin instead of 32)."""

    backend = "cupy-cuda-c"
    variant = "msc"
    dims = (2,)
    algorithms = ("metropolis",)
    layout_multiple = 128
    stream = "msc"
    bytes_per_update = 3.0 / 8.0
    lazy = 1

    def _setup(self) -> None:
        self.nw = self.lh // 64
        self.ly = self.shape[0]
        defs = (("NW", self.nw), ("LY", self.ly), ("LAZY", self.lazy))
        self.kernel = _module(defs).get_function("msc_half")
        self.w = [cp.zeros((self.rows, self.nw), dtype=cp.uint64) for _ in range(2)]
        self.k0, self.k1 = split_seed(self.seed)
        self.n_words = self.rows * self.nw
        self.stats: cp.ndarray | None = None

    def _half_sweep(self, color: int) -> None:
        blocks = (self.n_words + THREADS - 1) // THREADS
        stats = self.stats if self.stats is not None else np.uint64(0)
        self.kernel((blocks,), (THREADS,),
                    (self.w[color], self.w[1 - color], np.int64(self.n_words), np.uint32(self.t),
                     np.uint32(color), np.uint32(self.k0), np.uint32(self.k1),
                     np.uint32(self.table[3]), np.uint32(self.table[4]), stats))

    def philox_calls_per_word(self, sweeps: int = 2, warp: bool = False) -> float:
        """Mean Philox calls per word (diagnostic; slows the kernel). ``warp=True`` returns the
        per-warp maximum instead, i.e. the calls the SIMT hardware actually executes."""
        self.stats = cp.zeros(2, dtype=cp.uint64)
        self.sweep(sweeps)
        calls = int(self.stats.get()[1 if warp else 0])
        self.stats = None
        return calls / (2 * sweeps * self.n_words)

    def get_lattice(self) -> np.ndarray:
        w0, w1 = cp.asnumpy(self.w[0]), cp.asnumpy(self.w[1])
        return from_compressed(unpack_bits(w0), unpack_bits(w1), self.shape)

    def set_lattice(self, lat: np.ndarray) -> None:
        c0, c1 = to_compressed(np.asarray(lat, dtype=np.int8).reshape(self.shape))
        self.w = [cp.asarray(pack_bits(c0)), cp.asarray(pack_bits(c1))]

    def _fill_cold(self) -> None:
        self.w = [cp.full((self.rows, self.nw), np.uint64(0xFFFFFFFFFFFFFFFF), dtype=cp.uint64)
                  for _ in range(2)]

    def _fill_hot(self) -> None:
        for c, arr in enumerate(self.w):
            _hot_u64(np.uint64(2 * self.seed + c), arr)

    def synchronize(self) -> None:
        cp.cuda.Device().synchronize()

    def observables(self) -> tuple[float, float]:
        w = [_as_torch(x.view(cp.int64)) for x in self.w]
        return observables_packed(w[0], w[1], self.shape)

    def magnetization(self) -> float:
        return self.observables()[0]

    def energy(self) -> float:
        return self.observables()[1]


class CupyMultiSpinEager(CupyMultiSpin):
    """Ablation: same results, but always draws all 32 bits per spin (16 Philox calls/word)."""

    variant = "msc-eager32"
    lazy = 0
