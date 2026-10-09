"""NVIDIA Warp backends (kernels in Python, compiled by Warp to CUDA)."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from ..lattice import from_compressed, pack_bits, to_compressed, unpack_bits
from ..philox import split_seed
from .base import IsingSimulation
from .observables_torch import observables_int8, observables_packed

wp.config.quiet = True
wp.init()

vec4u = wp.types.vector(length=4, dtype=wp.uint32)


@wp.func
def philox4x32_10(c0: wp.uint32, c1: wp.uint32, c2: wp.uint32, c3: wp.uint32, k0: wp.uint32,
                  k1: wp.uint32) -> vec4u:
    for _r in range(10):
        p0 = wp.uint64(c0) * wp.uint64(0xD2511F53)
        p1 = wp.uint64(c2) * wp.uint64(0xCD9E8D57)
        n0 = wp.uint32(p1 >> wp.uint64(32)) ^ c1 ^ k0
        n2 = wp.uint32(p0 >> wp.uint64(32)) ^ c3 ^ k1
        c1 = wp.uint32(p1 & wp.uint64(0xFFFFFFFF))
        c3 = wp.uint32(p0 & wp.uint64(0xFFFFFFFF))
        c0 = n0
        c2 = n2
        k0 = k0 + wp.uint32(0x9E3779B9)
        k1 = k1 + wp.uint32(0xBB67AE85)
    return vec4u(c0, c1, c2, c3)


@wp.func
def site_update(s: int, h: int, u: wp.uint32, table: wp.array(dtype=wp.uint32), ndim: int,
                heatbath: int) -> wp.int8:
    if heatbath != 0:
        if u < table[(h + 2 * ndim) // 2]:
            return wp.int8(1)
        return wp.int8(-1)
    x = s * h
    if x <= 0 or u < table[(x + 2 * ndim) // 2]:
        return wp.int8(-s)
    return wp.int8(s)


@wp.kernel
def int8_half(own: wp.array(dtype=wp.int8), other: wp.array(dtype=wp.int8),
              table: wp.array(dtype=wp.uint32), lh: int, ly: int, lz: int, ndim: int,
              heatbath: int, t: wp.uint32, color: int, k0: wp.uint32, k1: wp.uint32):
    z, y, qj = wp.tid()
    j = qj * 4
    row = z * ly + y
    n = row * lh + j
    p = (y + z + color) & 1
    yu = y + 1
    if yu == ly:
        yu = 0
    yd = y - 1
    if yd < 0:
        yd = ly - 1
    zu = z + 1
    if zu == lz:
        zu = 0
    zd = z - 1
    if zd < 0:
        zd = lz - 1
    q = wp.uint32(n // 4)
    r = philox4x32_10(q, t, wp.uint32(color), wp.uint32(0), k0, k1)
    for lane in range(4):
        jj = j + lane
        h = int(other[n + lane])
        h += int(other[(z * ly + yu) * lh + jj]) + int(other[(z * ly + yd) * lh + jj])
        if ndim == 3:
            h += int(other[(zu * ly + y) * lh + jj]) + int(other[(zd * ly + y) * lh + jj])
        if p == 1:
            jn = jj + 1
            if jn == lh:
                jn = 0
            h += int(other[row * lh + jn])
        else:
            jn = jj - 1
            if jn < 0:
                jn = lh - 1
            h += int(other[row * lh + jn])
        own[n + lane] = site_update(int(own[n + lane]), h, r[lane], table, ndim, heatbath)


@wp.func
def lazy_step(r: wp.uint64, bit: int, thr4: wp.uint32, thr8: wp.uint32, a0: wp.uint64,
              a1: wp.uint64, acc: wp.uint64, und: wp.uint64) -> wp.vec2ul:
    t4 = wp.uint64(0)
    t8 = wp.uint64(0)
    if ((thr4 >> wp.uint32(bit)) & wp.uint32(1)) != wp.uint32(0):
        t4 = ~wp.uint64(0)
    if ((thr8 >> wp.uint32(bit)) & wp.uint32(1)) != wp.uint32(0):
        t8 = ~wp.uint64(0)
    tm = (a1 & t4) | (a0 & t8)
    acc = acc | (und & ~r & tm)
    und = und & ~(r ^ tm)
    return wp.vec2ul(acc, und)


@wp.kernel
def msc_half(own: wp.array(dtype=wp.uint64), other: wp.array(dtype=wp.uint64), nw: int, ly: int,
             t: wp.uint32, color: int, k0: wp.uint32, k1: wp.uint32, thr4: wp.uint32,
             thr8: wp.uint32):
    y, x = wp.tid()
    w = y * nw + x
    p = (y + color) & 1
    o = own[w]
    c = other[w]
    hz = wp.uint64(0)
    if p == 1:
        xn = x + 1
        if xn == nw:
            xn = 0
        hz = (c >> wp.uint64(1)) | (other[y * nw + xn] << wp.uint64(63))
    else:
        xp = x - 1
        if xp < 0:
            xp = nw - 1
        hz = (c << wp.uint64(1)) | (other[y * nw + xp] >> wp.uint64(63))
    yu = y + 1
    if yu == ly:
        yu = 0
    yd = y - 1
    if yd < 0:
        yd = ly - 1
    x1 = o ^ c
    x2 = o ^ hz
    x3 = o ^ other[yu * nw + x]
    x4 = o ^ other[yd * nw + x]
    anyb = x1 | x2 | x3 | x4
    a0 = ~anyb
    s1 = x1 ^ x2
    c1 = x1 & x2
    s2 = x3 ^ x4
    c2 = x3 & x4
    a2 = c1 | c2 | (s1 & s2)
    a1 = anyb & ~a2
    acc = wp.uint64(0)
    und = a0 | a1
    lp = int(0)
    while lp < 16 and und != wp.uint64(0):
        r = philox4x32_10(wp.uint32(w), t, wp.uint32(color), wp.uint32(1 + lp), k0, k1)
        v = lazy_step(wp.uint64(r[0]) | (wp.uint64(r[1]) << wp.uint64(32)), 31 - 2 * lp, thr4,
                      thr8, a0, a1, acc, und)
        v = lazy_step(wp.uint64(r[2]) | (wp.uint64(r[3]) << wp.uint64(32)), 30 - 2 * lp, thr4,
                      thr8, a0, a1, v[0], v[1])
        acc = v[0]
        und = v[1]
        lp += 1
    own[w] = o ^ (a2 | acc)


class _WarpBase(IsingSimulation):
    backend = "warp"

    def synchronize(self) -> None:
        wp.synchronize_device("cuda:0")

    def magnetization(self) -> float:
        return self.observables()[0]

    def energy(self) -> float:
        return self.observables()[1]


class WarpInt8(_WarpBase):
    """int8 compressed checkerboard; launch grid ``(Lz, Ly, Lx/8)``, 4 sites per thread."""

    variant = "int8"

    def _setup(self) -> None:
        self.ly = self.shape[-2]
        self.lz = self.shape[0] if self.ndim == 3 else 1
        self.c = [wp.ones(self.rows * self.lh, dtype=wp.int8, device="cuda:0") for _ in range(2)]
        self.k0, self.k1 = split_seed(self.seed)
        self._upload_table()

    def _upload_table(self) -> None:
        self.table_d = wp.array(self.table.astype(np.uint32), dtype=wp.uint32, device="cuda:0")

    def _half_sweep(self, color: int) -> None:
        wp.launch(int8_half, dim=(self.lz, self.ly, self.lh // 4), device="cuda:0",
                  inputs=[self.c[color], self.c[1 - color], self.table_d, self.lh, self.ly,
                          self.lz, self.ndim, int(self.algorithm == "heatbath"),
                          wp.uint32(self.t), color, wp.uint32(self.k0), wp.uint32(self.k1)])

    def get_lattice(self) -> np.ndarray:
        c0, c1 = (x.numpy().reshape(self.rows, self.lh) for x in self.c)
        return from_compressed(c0, c1, self.shape)

    def set_lattice(self, lat: np.ndarray) -> None:
        c0, c1 = to_compressed(np.asarray(lat, dtype=np.int8).reshape(self.shape))
        self.c = [wp.array(x.reshape(-1), dtype=wp.int8, device="cuda:0") for x in (c0, c1)]

    def observables(self) -> tuple[float, float]:
        c = [wp.to_torch(x).view(self.rows, self.lh) for x in self.c]
        return observables_int8(c[0], c[1], self.shape)


class WarpMultiSpin(_WarpBase):
    """Multi-spin coding (64 spins/word, lazy bitwise uniforms) written in Warp."""

    variant = "msc"
    dims = (2,)
    algorithms = ("metropolis",)
    layout_multiple = 128
    stream = "msc"
    bytes_per_update = 3.0 / 8.0

    def _setup(self) -> None:
        self.nw = self.lh // 64
        self.w = [wp.zeros(self.rows * self.nw, dtype=wp.uint64, device="cuda:0")
                  for _ in range(2)]
        self.k0, self.k1 = split_seed(self.seed)

    def _half_sweep(self, color: int) -> None:
        wp.launch(msc_half, dim=(self.rows, self.nw), device="cuda:0",
                  inputs=[self.w[color], self.w[1 - color], self.nw, self.shape[0],
                          wp.uint32(self.t), color, wp.uint32(self.k0), wp.uint32(self.k1),
                          wp.uint32(int(self.table[3])), wp.uint32(int(self.table[4]))])

    def get_lattice(self) -> np.ndarray:
        w0, w1 = (x.numpy().reshape(self.rows, self.nw) for x in self.w)
        return from_compressed(unpack_bits(w0), unpack_bits(w1), self.shape)

    def set_lattice(self, lat: np.ndarray) -> None:
        c0, c1 = to_compressed(np.asarray(lat, dtype=np.int8).reshape(self.shape))
        self.w = [wp.array(pack_bits(x).reshape(-1), dtype=wp.uint64, device="cuda:0")
                  for x in (c0, c1)]

    def observables(self) -> tuple[float, float]:
        w = [wp.to_torch(x).view(torch.int64).view(self.rows, self.nw) for x in self.w]
        return observables_packed(w[0], w[1], self.shape)
