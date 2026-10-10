"""Triton backends: int8 checkerboard and 64-spin multi-spin coding, Philox from Triton's
own ``philox_impl``."""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.random import philox_impl

from ..lattice import from_compressed, pack_bits, to_compressed, unpack_bits
from ..philox import split_seed
from .base import IsingSimulation
from .observables_torch import observables_int8, observables_packed


@triton.jit(do_not_specialize=["t", "color", "k0", "k1"])
def _int8_kernel(own_ptr, other_ptr, table_ptr, n_quads, t, color, k0, k1,
                 LH: tl.constexpr, LY: tl.constexpr, LZ: tl.constexpr, NDIM: tl.constexpr,
                 HEATBATH: tl.constexpr, IDX64: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if IDX64:
        q = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        q = pid * BLOCK + tl.arange(0, BLOCK)
    qmask = q < n_quads
    # one Philox4x32-10 call per 4 colour sites
    cq = q.to(tl.uint32)
    ct = tl.zeros_like(cq) + t.to(tl.uint32)
    cc = tl.zeros_like(cq) + color.to(tl.uint32)
    zero = tl.zeros_like(cq)
    r0, r1, r2, r3 = philox_impl(cq, ct, cc, zero, k0.to(tl.uint32), k1.to(tl.uint32))
    k = tl.arange(0, 4)[None, :]
    u = tl.where(k == 0, r0[:, None], tl.where(k == 1, r1[:, None],
                 tl.where(k == 2, r2[:, None], r3[:, None])))
    n = q[:, None] * 4 + k
    mask = qmask[:, None] & (k < 4)
    row = n // LH
    j = n - row * LH
    y = row % LY
    z = row // LY
    p = (y + z + color) & 1
    jn = tl.where(p == 1, j + 1, j - 1)
    jn = tl.where(jn < 0, jn + LH, tl.where(jn >= LH, jn - LH, jn))
    s = tl.load(own_ptr + n, mask=mask, other=1).to(tl.int32)
    h = tl.load(other_ptr + n, mask=mask, other=0).to(tl.int32)
    h += tl.load(other_ptr + row * LH + jn, mask=mask, other=0).to(tl.int32)
    yu = tl.where(y + 1 == LY, 0, y + 1)
    yd = tl.where(y == 0, LY - 1, y - 1)
    h += tl.load(other_ptr + (z * LY + yu) * LH + j, mask=mask, other=0).to(tl.int32)
    h += tl.load(other_ptr + (z * LY + yd) * LH + j, mask=mask, other=0).to(tl.int32)
    if NDIM == 3:
        zu = tl.where(z + 1 == LZ, 0, z + 1)
        zd = tl.where(z == 0, LZ - 1, z - 1)
        h += tl.load(other_ptr + (zu * LY + y) * LH + j, mask=mask, other=0).to(tl.int32)
        h += tl.load(other_ptr + (zd * LY + y) * LH + j, mask=mask, other=0).to(tl.int32)
    if HEATBATH:
        x = h
    else:
        x = s * h
    thr = tl.load(table_ptr + (x + 2 * NDIM) // 2, mask=mask, other=0)
    acc = u < thr
    if HEATBATH:
        new = tl.where(acc, 1, -1)
    else:
        new = tl.where(acc | (x <= 0), -s, s)
    tl.store(own_ptr + n, new.to(tl.int8), mask=mask)


class TritonInt8(IsingSimulation):
    """int8 compressed checkerboard, ``[BLOCK, 4]`` tiles, one Philox call per 4 sites."""

    backend = "triton"
    variant = "int8"
    block = 256
    force_idx64 = False  # tests exercise the 64-bit index path on small lattices

    def _setup(self) -> None:
        self.dev = torch.device("cuda")
        self.k0, self.k1 = split_seed(self.seed)
        self.n_quads = self.rows * self.lh // 4
        self.ly = self.shape[-2]
        self.lz = self.shape[0] if self.ndim == 3 else 1
        self._upload_table()

    def _upload_table(self) -> None:
        self.table_t = torch.as_tensor(self.table.view(np.int32), device="cuda")

    def _half_sweep(self, color: int) -> None:
        grid = (triton.cdiv(self.n_quads, self.block),)
        _int8_kernel[grid](self.c[color], self.c[1 - color], self.table_t, self.n_quads,
                           self.t, color, self.k0, self.k1, LH=self.lh, LY=self.ly, LZ=self.lz,
                           NDIM=self.ndim, HEATBATH=self.algorithm == "heatbath",
                           IDX64=self.force_idx64 or self.rows * self.lh >= 2**31 - 8,
                           BLOCK=self.block,
                           num_warps=4)

    def get_lattice(self) -> np.ndarray:
        return from_compressed(self.c[0].cpu().numpy(), self.c[1].cpu().numpy(), self.shape)

    def set_lattice(self, lat: np.ndarray) -> None:
        c0, c1 = to_compressed(np.asarray(lat, dtype=np.int8).reshape(self.shape))
        self.c = [torch.as_tensor(c0, device=self.dev), torch.as_tensor(c1, device=self.dev)]

    def _fill_hot(self) -> None:
        g = torch.Generator(device="cuda").manual_seed(self.seed & 0x7FFFFFFFFFFFFFFF)
        self.c = [torch.randint(0, 2, (self.rows, self.lh), generator=g, device="cuda",
                                dtype=torch.int8).mul_(2).sub_(1) for _ in range(2)]

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def observables(self) -> tuple[float, float]:
        return observables_int8(self.c[0], self.c[1], self.shape)

    def magnetization(self) -> float:
        return self.observables()[0]

    def energy(self) -> float:
        return self.observables()[1]


@triton.jit
def _lazy_step(r, bit, thr4, thr8, a0, a1, acc, und):
    b4 = ((thr4 >> bit) & 1).to(tl.uint64)
    b8 = ((thr8 >> bit) & 1).to(tl.uint64)
    zero = tl.zeros_like(r)
    tm = (a1 & (zero - b4)) | (a0 & (zero - b8))
    acc = acc | (und & ~r & tm)
    und = und & ~(r ^ tm)
    return acc, und


@triton.jit(do_not_specialize=["t", "color", "k0", "k1", "thr4", "thr8"])
def _msc_kernel(own_ptr, other_ptr, n_words, t, color, k0, k1, thr4, thr8,
                NW: tl.constexpr, LY: tl.constexpr, BLOCK: tl.constexpr):
    w = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = w < n_words
    y = w // NW
    x = w - y * NW
    p = (y + color) & 1
    o = tl.load(own_ptr + w, mask=mask, other=0)
    c = tl.load(other_ptr + w, mask=mask, other=0)
    xn = tl.where(x + 1 == NW, 0, x + 1)
    xp = tl.where(x == 0, NW - 1, x - 1)
    nb = tl.load(other_ptr + y * NW + tl.where(p == 1, xn, xp), mask=mask, other=0)
    hz = tl.where(p == 1, (c >> 1) | (nb << 63), (c << 1) | (nb >> 63))
    yu = tl.where(y + 1 == LY, 0, y + 1)
    yd = tl.where(y == 0, LY - 1, y - 1)
    x1 = o ^ c
    x2 = o ^ hz
    x3 = o ^ tl.load(other_ptr + yu * NW + x, mask=mask, other=0)
    x4 = o ^ tl.load(other_ptr + yd * NW + x, mask=mask, other=0)
    anyb = x1 | x2 | x3 | x4
    a0 = ~anyb
    s1 = x1 ^ x2
    c1 = x1 & x2
    s2 = x3 ^ x4
    c2 = x3 & x4
    a2 = c1 | c2 | (s1 & s2)
    a1 = anyb & ~a2
    acc = tl.zeros_like(o)
    und = tl.where(mask, a0 | a1, 0)
    cw = w.to(tl.uint32)
    ct = tl.zeros_like(cw) + t.to(tl.uint32)
    cc = tl.zeros_like(cw) + color.to(tl.uint32)
    t4 = thr4.to(tl.uint32)
    t8 = thr8.to(tl.uint32)
    lp = 0
    # lazy: stop as soon as every lane of the block is decided
    while (lp < 16) & (tl.max((und != 0).to(tl.int32), axis=0) > 0):
        r0, r1, r2, r3 = philox_impl(cw, ct, cc, tl.zeros_like(cw) + (1 + lp).to(tl.uint32),
                                     k0.to(tl.uint32), k1.to(tl.uint32))
        acc, und = _lazy_step(r0.to(tl.uint64) | (r1.to(tl.uint64) << 32), 31 - 2 * lp,
                              t4, t8, a0, a1, acc, und)
        acc, und = _lazy_step(r2.to(tl.uint64) | (r3.to(tl.uint64) << 32), 30 - 2 * lp,
                              t4, t8, a0, a1, acc, und)
        lp += 1
    tl.store(own_ptr + w, o ^ (a2 | acc), mask=mask)


class TritonMultiSpin(TritonInt8):
    """Multi-spin coding in Triton: uint64 tiles, bit-sliced counts, block-lazy uniforms."""

    variant = "msc"
    dims = (2,)
    algorithms = ("metropolis",)
    layout_multiple = 128
    stream = "msc"
    bytes_per_update = 3.0 / 8.0
    block = 64

    def _setup(self) -> None:
        self.dev = torch.device("cuda")
        self.nw = self.lh // 64
        self.n_words = self.rows * self.nw
        self.k0, self.k1 = split_seed(self.seed)

    def _upload_table(self) -> None:
        pass

    def _half_sweep(self, color: int) -> None:
        grid = (triton.cdiv(self.n_words, self.block),)
        _msc_kernel[grid](self.w[color], self.w[1 - color], self.n_words, self.t, color,
                          self.k0, self.k1, int(self.table[3]), int(self.table[4]),
                          NW=self.nw, LY=self.shape[0], BLOCK=self.block,
                          num_warps=max(1, self.block // 32))

    def get_lattice(self) -> np.ndarray:
        w0, w1 = (x.view(torch.int64).cpu().numpy().view(np.uint64) for x in self.w)
        return from_compressed(unpack_bits(w0), unpack_bits(w1), self.shape)

    def set_lattice(self, lat: np.ndarray) -> None:
        c0, c1 = to_compressed(np.asarray(lat, dtype=np.int8).reshape(self.shape))
        self.w = [torch.as_tensor(pack_bits(x).view(np.int64), device=self.dev)
                  .view(torch.uint64) for x in (c0, c1)]

    def _fill_cold(self) -> None:
        self.w = [torch.full((self.rows, self.nw), -1, dtype=torch.int64, device=self.dev)
                  .view(torch.uint64) for _ in range(2)]

    def _fill_hot(self) -> None:
        g = torch.Generator(device="cuda").manual_seed(self.seed & 0x7FFFFFFFFFFFFFFF)
        self.w = [torch.randint(-2**63, 2**63 - 1, (self.rows, self.nw), generator=g,
                                device="cuda", dtype=torch.int64).view(torch.uint64)
                  for _ in range(2)]

    def observables(self) -> tuple[float, float]:
        w = [x.view(torch.int64) for x in self.w]
        return observables_packed(w[0], w[1], self.shape)
