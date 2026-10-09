"""Pure NumPy reference backends (the correctness oracle; also what CI runs)."""

from __future__ import annotations

import numpy as np

from ..lattice import from_compressed, pack_bits, row_parity, to_compressed, unpack_bits
from ..philox import site_uniforms, word_randoms
from .base import IsingSimulation

ALL_ONES = np.uint64(0xFFFFFFFFFFFFFFFF)


class NumpyCheckerboard(IsingSimulation):
    """int8 spins in the compressed checkerboard layout, one Philox word per site."""

    backend = "numpy"
    variant = "int8"
    gpu = False

    def _setup(self) -> None:
        self.c = [np.ones((self.rows, self.lh), np.int8), np.ones((self.rows, self.lh), np.int8)]
        self.rp = row_parity(self.shape)[:, None]
        self.grid = self.shape[:-1] + (self.lh,)

    def _neighbor_sum(self, color: int) -> np.ndarray:
        other = self.c[1 - color]
        p = (self.rp + color) % 2
        h = other + np.where(p == 1, np.roll(other, -1, axis=1), np.roll(other, 1, axis=1))
        g = other.reshape(self.grid)
        for ax in range(self.ndim - 1):
            h = h + (np.roll(g, 1, axis=ax) + np.roll(g, -1, axis=ax)).reshape(self.rows, self.lh)
        return h.astype(np.int8)

    def _half_sweep(self, color: int) -> None:
        s = self.c[color]
        h = self._neighbor_sum(color).astype(np.int64)
        u = site_uniforms(self.rows * self.lh, self.t, color, self.seed).reshape(s.shape)
        x = h if self.algorithm == "heatbath" else s * h
        acc = u < self.table[(x + 2 * self.ndim) // 2]
        if self.algorithm == "heatbath":
            self.c[color] = np.where(acc, 1, -1).astype(np.int8)
        else:
            self.c[color] = np.where(acc | (x <= 0), -s, s).astype(np.int8)

    def get_lattice(self) -> np.ndarray:
        return from_compressed(self.c[0], self.c[1], self.shape)

    def set_lattice(self, lat: np.ndarray) -> None:
        c0, c1 = to_compressed(np.asarray(lat, dtype=np.int8).reshape(self.shape))
        self.c = [c0, c1]


def msc_flip_mask(
    own: np.ndarray, nbs: list[np.ndarray], thr4: int, thr8: int, rand_level
) -> np.ndarray:
    """Bit-sliced 2D Metropolis decision for 64 spins per word.

    ``x_k = own ^ nb_k`` marks anti-aligned neighbours; with ``a`` of them ``dE = 8 - 4a``.
    Lanes with ``a >= 2`` always flip; lanes with ``a = 1`` (``a = 0``) flip iff the lane's
    uniform ``u < thr4`` (``thr8``). ``u`` is built MSB-first from bit ``b`` of the random words
    of successive levels, and the comparison is done lazily, all 64 lanes at once.
    """
    x1, x2, x3, x4 = (own ^ nb for nb in nbs)
    any_anti = x1 | x2 | x3 | x4
    a0 = ~any_anti
    s1, c1 = x1 ^ x2, x1 & x2
    s2, c2 = x3 ^ x4, x3 & x4
    a2 = c1 | c2 | (s1 & s2)
    a1 = any_anti & ~a2
    acc = np.zeros_like(own)
    und = a0 | a1
    for k in range(32):
        if not und.any():
            break
        r = rand_level(k)
        bit = 31 - k
        t4 = ALL_ONES if (thr4 >> bit) & 1 else np.uint64(0)
        t8 = ALL_ONES if (thr8 >> bit) & 1 else np.uint64(0)
        tm = (a1 & t4) | (a0 & t8)
        acc |= und & ~r & tm
        und &= ~(r ^ tm)
    return a2 | acc


class NumpyMultiSpin(IsingSimulation):
    """Reference for multi-spin coding: 64 spins per uint64, bitwise neighbour logic (2D)."""

    backend = "numpy"
    variant = "msc"
    dims = (2,)
    algorithms = ("metropolis",)
    layout_multiple = 128
    stream = "msc"
    bytes_per_update = 3.0 / 8.0
    gpu = False

    def _setup(self) -> None:
        self.nw = self.lh // 64
        self.w = [np.zeros((self.rows, self.nw), np.uint64) for _ in range(2)]
        self.rp = row_parity(self.shape)[:, None]

    def neighbors(self, color: int) -> list[np.ndarray]:
        o = self.w[1 - color]
        one, s63 = np.uint64(1), np.uint64(63)
        right = (o >> one) | (np.roll(o, -1, axis=1) << s63)  # j + 1
        left = (o << one) | (np.roll(o, 1, axis=1) >> s63)  # j - 1
        p = (self.rp + color) % 2
        horiz = np.where(p == 1, right, left)
        return [o, horiz, np.roll(o, 1, axis=0), np.roll(o, -1, axis=0)]

    def _half_sweep(self, color: int) -> None:
        own = self.w[color]
        n_words = self.rows * self.nw
        cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def rand_level(k: int) -> np.ndarray:
            if k // 2 not in cache:
                cache[k // 2] = word_randoms(n_words, self.t, color, self.seed, k // 2)
            return cache[k // 2][k % 2].reshape(own.shape)

        flip = msc_flip_mask(own, self.neighbors(color), int(self.table[3]), int(self.table[4]),
                             rand_level)
        self.w[color] = own ^ flip

    def get_lattice(self) -> np.ndarray:
        return from_compressed(unpack_bits(self.w[0]), unpack_bits(self.w[1]), self.shape)

    def set_lattice(self, lat: np.ndarray) -> None:
        c0, c1 = to_compressed(np.asarray(lat, dtype=np.int8).reshape(self.shape))
        self.w = [pack_bits(c0), pack_bits(c1)]
