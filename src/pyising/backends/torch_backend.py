"""PyTorch backends: eager tensor ops and the same code under ``torch.compile``."""

from __future__ import annotations

import numpy as np
import torch

from ..lattice import from_compressed, row_parity, to_compressed
from ..philox import M0, M1, MASK32, ROUNDS, W0, W1, split_seed
from .base import IsingSimulation
from .observables_torch import observables_int8


def philox_torch(c0, c1, c2, c3, k0: int, k1: int, rounds: int = ROUNDS):
    """Philox4x32-10 on int64 tensors holding uint32 values (64-bit products wrap mod 2^64)."""
    for r in range(rounds):
        p0 = c0 * M0
        p1 = c2 * M1
        c0, c1, c2, c3 = (
            ((p1 >> 32) & MASK32) ^ c1 ^ k0,
            p1 & MASK32,
            ((p0 >> 32) & MASK32) ^ c3 ^ k1,
            p0 & MASK32,
        )
        if r != rounds - 1:
            k0 = (k0 + W0) & MASK32
            k1 = (k1 + W1) & MASK32
    return c0, c1, c2, c3


def half_sweep_torch(
    own: torch.Tensor,
    other: torch.Tensor,
    parity: torch.Tensor,
    table: torch.Tensor,
    t: torch.Tensor,
    q: torch.Tensor,
    color: int,
    k0: int,
    k1: int,
    grid: tuple[int, ...],
    heatbath: bool,
) -> torch.Tensor:
    """One colour update in the compressed layout; returns the new ``own`` array."""
    rows, lh = own.shape
    ndim = len(grid)
    right = torch.roll(other, -1, dims=1)
    left = torch.roll(other, 1, dims=1)
    h = other + torch.where(parity, right, left)
    g = other.view(grid)
    for ax in range(ndim - 1):
        h = h + (torch.roll(g, 1, dims=ax) + torch.roll(g, -1, dims=ax)).view(rows, lh)
    zero = torch.zeros_like(q)
    r = philox_torch(q, zero + t, zero + color, zero, k0, k1)
    u = torch.stack(r, dim=1).view(rows, lh)
    h64 = h.to(torch.int64)
    x = h64 if heatbath else own.to(torch.int64) * h64
    acc = u < table[(x + 2 * ndim) // 2]
    if heatbath:
        return torch.where(acc, 1, -1).to(torch.int8)
    return torch.where(acc | (x <= 0), -own, own)


class TorchEager(IsingSimulation):
    """Tensor-op implementation; dozens of small kernels per half sweep."""

    backend = "torch-eager"
    variant = "int8"

    def _fn(self):
        return half_sweep_torch

    def _setup(self) -> None:
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.c = [torch.ones((self.rows, self.lh), dtype=torch.int8, device=self.dev)
                  for _ in range(2)]
        rp = row_parity(self.shape)
        self.par = [torch.as_tensor((rp + c) % 2 == 1, device=self.dev)[:, None] for c in (0, 1)]
        self.grid = self.shape[:-1] + (self.lh,)
        self.k0, self.k1 = split_seed(self.seed)
        # quad index (one Philox call per 4 colour sites); an input so Inductor keeps it int64
        self.q = torch.arange(self.rows * self.lh // 4, device=self.dev, dtype=torch.int64)
        self._upload_table()
        self.fn = self._fn()

    def _upload_table(self) -> None:
        self.table_t = torch.as_tensor(self.table.astype(np.int64), device=self.dev)

    def _half_sweep(self, color: int) -> None:
        t = torch.tensor(self.t, dtype=torch.int64, device=self.dev)
        self.c[color] = self.fn(self.c[color], self.c[1 - color], self.par[color], self.table_t,
                                t, self.q, color, self.k0, self.k1, self.grid,
                                self.algorithm == "heatbath")

    def get_lattice(self) -> np.ndarray:
        return from_compressed(self.c[0].cpu().numpy(), self.c[1].cpu().numpy(), self.shape)

    def set_lattice(self, lat: np.ndarray) -> None:
        c0, c1 = to_compressed(np.asarray(lat, dtype=np.int8).reshape(self.shape))
        self.c = [torch.as_tensor(c0, device=self.dev), torch.as_tensor(c1, device=self.dev)]

    def synchronize(self) -> None:
        if self.dev.type == "cuda":
            torch.cuda.synchronize()

    def observables(self) -> tuple[float, float]:
        return observables_int8(self.c[0], self.c[1], self.shape)

    def magnetization(self) -> float:
        return self.observables()[0]

    def energy(self) -> float:
        return self.observables()[1]


class TorchCompile(TorchEager):
    """Same Python as :class:`TorchEager`, fused by ``torch.compile`` (Inductor -> Triton)."""

    backend = "torch-compile"
    variant = "int8"

    def _fn(self):
        return torch.compile(half_sweep_torch, dynamic=False, fullgraph=True)
