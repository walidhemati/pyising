"""Numba-CUDA backends (kernels written in Python, compiled by Numba to PTX)."""

from __future__ import annotations

import functools

import numpy as np

from ._gpu import prepare_numba

prepare_numba()
import torch  # noqa: E402
from numba import cuda, int32, uint32, uint64  # noqa: E402

from ..lattice import from_compressed, pack_bits, to_compressed, unpack_bits  # noqa: E402
from ..philox import M0, M1, W0, W1, split_seed  # noqa: E402
from .base import IsingSimulation  # noqa: E402
from .observables_torch import observables_int8, observables_packed  # noqa: E402

_M0, _M1 = np.uint64(M0), np.uint64(M1)
_W0, _W1 = np.uint32(W0), np.uint32(W1)
_S32 = np.uint64(32)
_ONE64 = np.uint64(1)
_S63 = np.uint64(63)
_ALL = np.uint64(0xFFFFFFFFFFFFFFFF)
THREADS = 256


@cuda.jit(device=True, inline=True)
def philox4x32_10(c0, c1, c2, c3, k0, k1):
    """Philox4x32-10 on uint32 values (64-bit products give hi/lo words)."""
    for _ in range(10):
        p0 = uint64(c0) * _M0
        p1 = uint64(c2) * _M1
        n0 = uint32(p1 >> _S32) ^ c1 ^ k0
        n2 = uint32(p0 >> _S32) ^ c3 ^ k1
        c1 = uint32(p1)
        c3 = uint32(p0)
        c0 = uint32(n0)
        c2 = uint32(n2)
        k0 = uint32(k0 + _W0)
        k1 = uint32(k1 + _W1)
    return c0, c1, c2, c3


@functools.cache
def _int8_kernel(LH: int, LY: int, LZ: int, NDIM: int, HEATBATH: bool):
    """4 int8 spins per thread, moved as one 32-bit word (the ``char4`` of the CUDA C code)."""
    two_d = 2 * NDIM
    NQ = LH // 4

    @cuda.jit(device=True, inline=True)
    def sbyte(word, k):
        b = int32((word >> uint32(8 * k)) & uint32(0xFF))
        return b - int32(256) if b >= int32(128) else b

    @cuda.jit(device=True, inline=True)
    def update(s, h, u, table):
        if HEATBATH:
            return int32(1) if u < table[int32(h + two_d) // int32(2)] else int32(-1)
        x = int32(s * h)
        return -s if (x <= 0 or u < table[(x + int32(two_d)) // int32(2)]) else s

    @cuda.jit
    def kernel(own, other, table, n_quads, t, color, k0, k1):
        q = cuda.grid(1)
        if q >= n_quads:
            return
        row = q // NQ
        jq = q - row * NQ
        y = row % LY
        z = row // LY
        p = (y + z + color) & 1
        rb = row * NQ
        c = other[q]
        up = other[(z * LY + (0 if y + 1 == LY else y + 1)) * NQ + jq]
        dn = other[(z * LY + (LY - 1 if y == 0 else y - 1)) * NQ + jq]
        if NDIM == 3:
            zu = other[((0 if z + 1 == LZ else z + 1) * LY + y) * NQ + jq]
            zd = other[((LZ - 1 if z == 0 else z - 1) * LY + y) * NQ + jq]
        if p == 1:
            nxt = other[rb + (0 if jq + 1 == NQ else jq + 1)]
        else:
            prv = other[rb + (NQ - 1 if jq == 0 else jq - 1)]
        r0, r1, r2, r3 = philox4x32_10(uint32(q), uint32(t), uint32(color), uint32(0),
                                       uint32(k0), uint32(k1))
        s4 = own[q]
        out = uint32(0)
        for lane in range(4):
            h = sbyte(c, lane) + sbyte(up, lane) + sbyte(dn, lane)
            if NDIM == 3:
                h += sbyte(zu, lane) + sbyte(zd, lane)
            if p == 1:
                h += sbyte(nxt, 0) if lane == 3 else sbyte(c, lane + 1)
            else:
                h += sbyte(prv, 3) if lane == 0 else sbyte(c, lane - 1)
            u = r0 if lane == 0 else (r1 if lane == 1 else (r2 if lane == 2 else r3))
            v = update(sbyte(s4, lane), h, u, table)
            out |= (uint32(v) & uint32(0xFF)) << uint32(8 * lane)
        own[q] = out

    return kernel


@functools.cache
def _msc_kernel(NW: int, LY: int):
    @cuda.jit(device=True, inline=True)
    def step(r, bit, thr4, thr8, a0, a1, acc, und):
        t4 = _ALL if (thr4 >> bit) & 1 else uint64(0)
        t8 = _ALL if (thr8 >> bit) & 1 else uint64(0)
        tm = (a1 & t4) | (a0 & t8)
        acc |= und & ~r & tm
        und &= ~(r ^ tm)
        return acc, und

    @cuda.jit
    def kernel(own, other, n_words, t, color, k0, k1, thr4, thr8):
        w = cuda.grid(1)
        if w >= n_words:
            return
        y = w // NW
        x = w - y * NW
        rb = y * NW
        p = (y + color) & 1
        o = own[w]
        c = other[w]
        if p == 1:
            nx = other[rb + (0 if x + 1 == NW else x + 1)]
            hz = (c >> _ONE64) | (nx << _S63)
        else:
            pv = other[rb + (NW - 1 if x == 0 else x - 1)]
            hz = (c << _ONE64) | (pv >> _S63)
        up = other[(0 if y + 1 == LY else y + 1) * NW + x]
        dn = other[(LY - 1 if y == 0 else y - 1) * NW + x]
        x1 = o ^ c
        x2 = o ^ hz
        x3 = o ^ up
        x4 = o ^ dn
        anyb = x1 | x2 | x3 | x4
        a0 = ~anyb
        s1 = x1 ^ x2
        c1 = x1 & x2
        s2 = x3 ^ x4
        c2 = x3 & x4
        a2 = c1 | c2 | (s1 & s2)
        a1 = anyb & ~a2
        acc = uint64(0)
        und = a0 | a1
        t4u, t8u = uint32(thr4), uint32(thr8)
        lp = 0
        while lp < 16 and und != uint64(0):
            r0, r1, r2, r3 = philox4x32_10(uint32(w), uint32(t), uint32(color),
                                           uint32(1 + lp), uint32(k0), uint32(k1))
            acc, und = step(uint64(r0) | (uint64(r1) << _S32), 31 - 2 * lp, t4u, t8u, a0, a1,
                            acc, und)
            acc, und = step(uint64(r2) | (uint64(r3) << _S32), 30 - 2 * lp, t4u, t8u, a0, a1,
                            acc, und)
            lp += 1
        own[w] = o ^ (a2 | acc)

    return kernel


class _NumbaBase(IsingSimulation):
    backend = "numba"

    def synchronize(self) -> None:
        cuda.synchronize()

    def magnetization(self) -> float:
        return self.observables()[0]

    def energy(self) -> float:
        return self.observables()[1]


class NumbaInt8(_NumbaBase):
    """int8 compressed checkerboard; one thread per 4 sites (one 32-bit word), one Philox call."""

    variant = "int8"

    def _setup(self) -> None:
        self.ly = self.shape[-2]
        self.lz = self.shape[0] if self.ndim == 3 else 1
        self.kernel = _int8_kernel(self.lh, self.ly, self.lz, self.ndim,
                                   self.algorithm == "heatbath")
        self.c = [cuda.to_device(np.ones(self.rows * self.lh, np.int8).view(np.uint32))
                  for _ in range(2)]
        self.k0, self.k1 = split_seed(self.seed)
        self.n_quads = self.rows * self.lh // 4
        self._upload_table()

    def _upload_table(self) -> None:
        self.table_d = cuda.to_device(self.table.astype(np.uint32))

    def _half_sweep(self, color: int) -> None:
        blocks = (self.n_quads + THREADS - 1) // THREADS
        self.kernel[blocks, THREADS](self.c[color], self.c[1 - color], self.table_d,
                                     self.n_quads, self.t, color, self.k0, self.k1)

    def get_lattice(self) -> np.ndarray:
        c0, c1 = (x.copy_to_host().view(np.int8).reshape(self.rows, self.lh) for x in self.c)
        return from_compressed(c0, c1, self.shape)

    def set_lattice(self, lat: np.ndarray) -> None:
        c0, c1 = to_compressed(np.asarray(lat, dtype=np.int8).reshape(self.shape))
        self.c = [cuda.to_device(c0.reshape(-1).view(np.uint32)),
                  cuda.to_device(c1.reshape(-1).view(np.uint32))]

    def observables(self) -> tuple[float, float]:
        c = [torch.as_tensor(x.view(np.int32), device="cuda").view(torch.int8)
             .view(self.rows, self.lh) for x in self.c]
        return observables_int8(c[0], c[1], self.shape)


class NumbaMultiSpin(_NumbaBase):
    """Multi-spin coding (64 spins/word, lazy bitwise uniforms) written in Numba."""

    variant = "msc"
    dims = (2,)
    algorithms = ("metropolis",)
    layout_multiple = 128
    stream = "msc"
    bytes_per_update = 3.0 / 8.0

    def _setup(self) -> None:
        self.nw = self.lh // 64
        self.kernel = _msc_kernel(self.nw, self.shape[0])
        self.w = [cuda.to_device(np.zeros(self.rows * self.nw, np.uint64)) for _ in range(2)]
        self.k0, self.k1 = split_seed(self.seed)
        self.n_words = self.rows * self.nw

    def _half_sweep(self, color: int) -> None:
        blocks = (self.n_words + THREADS - 1) // THREADS
        self.kernel[blocks, THREADS](self.w[color], self.w[1 - color], self.n_words, self.t,
                                     color, self.k0, self.k1, int(self.table[3]),
                                     int(self.table[4]))

    def get_lattice(self) -> np.ndarray:
        w0, w1 = (x.copy_to_host().reshape(self.rows, self.nw) for x in self.w)
        return from_compressed(unpack_bits(w0), unpack_bits(w1), self.shape)

    def set_lattice(self, lat: np.ndarray) -> None:
        c0, c1 = to_compressed(np.asarray(lat, dtype=np.int8).reshape(self.shape))
        self.w = [cuda.to_device(pack_bits(c0).reshape(-1)),
                  cuda.to_device(pack_bits(c1).reshape(-1))]

    def observables(self) -> tuple[float, float]:
        w = [torch.as_tensor(x, device="cuda").view(torch.int64).view(self.rows, self.nw)
             for x in self.w]
        return observables_packed(w[0], w[1], self.shape)
