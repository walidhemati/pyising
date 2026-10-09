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

    def _setup(self) -> None:
        self.dev = torch.device("cuda")
        self.c = [torch.ones((self.rows, self.lh), dtype=torch.int8, device=self.dev)
                  for _ in range(2)]
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
                           IDX64=self.rows * self.lh >= 2**31 - 8, BLOCK=self.block,
                           num_warps=4)

    def get_lattice(self) -> np.ndarray:
        return from_compressed(self.c[0].cpu().numpy(), self.c[1].cpu().numpy(), self.shape)

    def set_lattice(self, lat: np.ndarray) -> None:
        c0, c1 = to_compressed(np.asarray(lat, dtype=np.int8).reshape(self.shape))
        self.c = [torch.as_tensor(c0, device=self.dev), torch.as_tensor(c1, device=self.dev)]

    def _fill_hot(self) -> None:
        g = torch.Generator(device="cuda").manual_seed(self.seed & 0x7FFFFFFFFFFFFFFF)
        self.c = [(torch.randint(0, 2, (self.rows, self.lh), generator=g, device="cuda",
                                 dtype=torch.int8) * 2 - 1) for _ in range(2)]

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def observables(self) -> tuple[float, float]:
        return observables_int8(self.c[0], self.c[1], self.shape)

    def magnetization(self) -> float:
        return self.observables()[0]

    def energy(self) -> float:
        return self.observables()[1]
