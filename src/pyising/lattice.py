"""Lattice layouts, observables and acceptance tables shared by all backends.

Conventions
-----------
* A configuration is an ``int8`` array of +-1 with shape ``(L, L)`` (2D) or ``(L, L, L)`` (3D),
  periodic boundaries, coupling ``J = 1``, no field: ``H = -sum_<ij> s_i s_j``.
* Site ``(..., y, x)`` has colour ``c = (sum of indices) % 2``. A sweep updates colour 0, then 1.
* Compressed ("checkerboard") layout: colour ``c`` is stored as an array of shape
  ``(rows, Lx // 2)`` where ``rows`` flattens all but the last axis. In row ``r`` the colour-``c``
  sites sit at ``x = 2 j + p`` with ``p = (row-index-sum + c) % 2``. The colour-site index used
  for the random stream is ``n = r * (Lx // 2) + j``, which is the same in every layout.
* Packed layout (multi-spin coding): the compressed colour array with 64 consecutive ``j`` per
  ``uint64`` word, bit ``b`` of word ``w`` holding ``j = 64 w + b``; bit 1 means spin +1.
"""

from __future__ import annotations

import math

import numpy as np

UINT32_MAX = 0xFFFFFFFF


def check_shape(shape: tuple[int, ...], multiple: int = 8) -> tuple[int, ...]:
    """Validate a lattice shape: 2D or 3D, every extent even, ``Lx`` a multiple of ``multiple``."""
    shape = tuple(int(s) for s in shape)
    if len(shape) not in (2, 3):
        raise ValueError(f"only 2D and 3D lattices are supported, got shape {shape}")
    if any(s < 2 or s % 2 for s in shape):
        raise ValueError(f"all extents must be even and >= 2, got {shape}")
    if shape[-1] % multiple:
        raise ValueError(f"last extent must be a multiple of {multiple}, got {shape[-1]}")
    return shape


def parity_grid(shape: tuple[int, ...]) -> np.ndarray:
    """``(sum of indices) % 2`` for every site, as uint8."""
    idx = np.indices(shape, dtype=np.int64).sum(axis=0)
    return (idx % 2).astype(np.uint8)


def row_parity(shape: tuple[int, ...]) -> np.ndarray:
    """Parity of the sum of the non-x indices for each flattened row (length ``rows``)."""
    rows_shape = shape[:-1]
    return (np.indices(rows_shape).sum(axis=0) % 2).reshape(-1).astype(np.int64)


def to_compressed(lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a full lattice into the two compressed colour arrays ``(c0, c1)``."""
    shape = lat.shape
    lx = shape[-1]
    flat = lat.reshape(-1, lx)
    rp = row_parity(shape)
    out = []
    for c in (0, 1):
        p = (rp + c) % 2
        cols = 2 * np.arange(lx // 2)[None, :] + p[:, None]
        out.append(np.ascontiguousarray(np.take_along_axis(flat, cols, axis=1)))
    return out[0], out[1]


def from_compressed(c0: np.ndarray, c1: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Inverse of :func:`to_compressed`."""
    lx = shape[-1]
    flat = np.empty((c0.shape[0], lx), dtype=c0.dtype)
    rp = row_parity(shape)
    for c, arr in ((0, c0), (1, c1)):
        p = (rp + c) % 2
        cols = 2 * np.arange(lx // 2)[None, :] + p[:, None]
        np.put_along_axis(flat, cols, arr, axis=1)
    return flat.reshape(shape)


def pack_bits(comp: np.ndarray) -> np.ndarray:
    """Compressed int8 colour array ``(rows, Lh)`` -> packed uint64 ``(rows, Lh // 64)``."""
    rows, lh = comp.shape
    if lh % 64:
        raise ValueError("Lx/2 must be a multiple of 64 for the packed layout")
    bits = (comp > 0).astype(np.uint64).reshape(rows, lh // 64, 64)
    weights = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))
    return np.bitwise_or.reduce(bits * weights, axis=2)


def unpack_bits(words: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_bits`, returns int8 +-1."""
    rows, nw = words.shape
    shifts = np.arange(64, dtype=np.uint64)
    bits = (words[:, :, None] >> shifts) & np.uint64(1)
    return (bits.astype(np.int8) * 2 - 1).reshape(rows, nw * 64)


def random_lattice(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Infinite-temperature start: i.i.d. +-1 spins from ``numpy.random.default_rng(seed)``."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=shape, dtype=np.int8) * 2 - 1).astype(np.int8)


def neighbor_sum(lat: np.ndarray) -> np.ndarray:
    """Sum of the 2d nearest neighbours of every site (periodic), as int8."""
    h = np.zeros_like(lat, dtype=np.int8)
    for ax in range(lat.ndim):
        h += np.roll(lat, 1, axis=ax)
        h += np.roll(lat, -1, axis=ax)
    return h


def energy_per_spin(lat: np.ndarray) -> float:
    """``E / N`` with ``E = -sum_<ij> s_i s_j``."""
    e = 0
    for ax in range(lat.ndim):
        e -= int(np.sum(lat.astype(np.int64) * np.roll(lat, 1, axis=ax)))
    return e / lat.size


def magnetization(lat: np.ndarray) -> float:
    """Mean spin ``M / N`` (signed)."""
    return float(np.sum(lat, dtype=np.int64)) / lat.size


def prob_to_threshold(p: float) -> int:
    """Map a probability to a uint32 threshold: accept iff ``u32 < threshold``."""
    if p >= 1.0:
        return UINT32_MAX
    return min(int(math.floor(p * 2.0**32)), UINT32_MAX)


def threshold_table(beta: float, ndim: int, algorithm: str) -> np.ndarray:
    """Acceptance table of length ``2 ndim + 1`` indexed by ``(X + 2 ndim) // 2``.

    Metropolis: ``X = s * h``; flip iff ``X <= 0`` or ``u < table[idx]`` (``p = exp(-2 beta X)``).
    Heat bath: ``X = h``; new spin is +1 iff ``u < table[idx]`` (``p = 1 / (1 + exp(-2 beta h))``).
    """
    z = 2 * ndim
    xs = range(-z, z + 1, 2)
    if algorithm == "metropolis":
        vals = [UINT32_MAX if x <= 0 else prob_to_threshold(math.exp(-2.0 * beta * x)) for x in xs]
    elif algorithm == "heatbath":
        vals = [prob_to_threshold(1.0 / (1.0 + math.exp(-2.0 * beta * x))) for x in xs]
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    return np.array(vals, dtype=np.uint32)
