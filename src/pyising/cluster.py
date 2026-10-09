"""Wolff single-cluster algorithm (CPU, Numba) for 2D/3D near-critical sampling.

Used for the Binder-cumulant crossings, where local updates suffer critical slowing down.
Falls back to a slow pure-Python loop if Numba is unavailable.
"""

from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - import guard
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore[no-redef]
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


def neighbor_table(shape: tuple[int, ...]) -> np.ndarray:
    """``(N, 2 d)`` int32 table of periodic nearest neighbours (C order)."""
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    cols = []
    for ax in range(len(shape)):
        cols.append(np.roll(idx, 1, axis=ax).reshape(-1))
        cols.append(np.roll(idx, -1, axis=ax).reshape(-1))
    return np.stack(cols, axis=1).astype(np.int32)


@njit(cache=True)
def _wolff(spins, nbr, p_add, n_clusters, seed):
    np.random.seed(seed)
    n = spins.size
    stack = np.empty(n, np.int32)
    flipped = 0
    for _ in range(n_clusters):
        i = np.random.randint(n)
        s0 = spins[i]
        spins[i] = -s0
        stack[0] = i
        top = 1
        while top > 0:
            top -= 1
            j = stack[top]
            for k in range(nbr.shape[1]):
                m = nbr[j, k]
                if spins[m] == s0 and np.random.random() < p_add:
                    spins[m] = -s0
                    stack[top] = m
                    top += 1
            flipped += 1
    return flipped


class Wolff:
    """Wolff cluster Monte Carlo on a periodic hypercubic lattice (``J = 1``)."""

    def __init__(self, shape: tuple[int, ...], beta: float, seed: int = 0) -> None:
        self.shape = tuple(shape)
        self.nbr = neighbor_table(self.shape)
        rng = np.random.default_rng(seed)
        self.spins = (rng.integers(0, 2, int(np.prod(shape)), dtype=np.int8) * 2 - 1).astype(
            np.int8)
        self.seed = int(seed) & 0x7FFFFFFF
        self.calls = 0
        self.clusters_per_sweep: int | None = None
        self.set_beta(beta)

    def set_beta(self, beta: float) -> None:
        self.beta = float(beta)
        self.p_add = 1.0 - math.exp(-2.0 * self.beta)

    def step(self, n_clusters: int) -> int:
        """Grow and flip ``n_clusters`` clusters; returns the number of spins flipped."""
        self.calls += 1
        return int(_wolff(self.spins, self.nbr, self.p_add, int(n_clusters),
                          (self.seed * 1000003 + self.calls) & 0x7FFFFFFF))

    def calibrate(self, n_clusters: int = 2000) -> None:
        """Estimate the mean cluster size and fix ``clusters_per_sweep`` (thermalises too)."""
        flipped = self.step(n_clusters)
        self.clusters_per_sweep = max(1, round(self.spins.size * n_clusters / max(flipped, 1)))

    def sweep(self, n_sweeps: int = 1) -> None:
        """Flip a *fixed* number of clusters worth about ``n_sweeps`` lattice sweeps.

        The count is frozen after :meth:`calibrate`; stopping on the number of flipped spins
        instead would bias measurements (the stopping time would depend on the state).
        """
        if self.clusters_per_sweep is None:
            self.calibrate()
        self.step(int(n_sweeps) * self.clusters_per_sweep)

    def magnetization(self) -> float:
        return float(self.spins.sum(dtype=np.int64)) / self.spins.size

    def energy(self) -> float:
        lat = self.spins.reshape(self.shape).astype(np.int64)
        e = 0
        for ax in range(lat.ndim):
            e -= int(np.sum(lat * np.roll(lat, 1, axis=ax)))
        return e / lat.size
