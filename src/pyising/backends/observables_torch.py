"""On-device magnetisation and energy from compressed (int8) or packed (uint64) colour arrays.

Every bond has exactly one colour-0 end, so ``E = -sum_{i in colour 0} s_i h_i``.
"""

from __future__ import annotations

import torch

from ..lattice import row_parity


def _neighbor_sum(other: torch.Tensor, shape: tuple[int, ...], color: int) -> torch.Tensor:
    rows, lh = other.shape
    par = torch.as_tensor((row_parity(shape) + color) % 2 == 1, device=other.device)[:, None]
    o = other.to(torch.int32)
    h = o + torch.where(par, torch.roll(o, -1, 1), torch.roll(o, 1, 1))
    g = o.view(shape[:-1] + (lh,))
    for ax in range(len(shape) - 1):
        h = h + (torch.roll(g, 1, ax) + torch.roll(g, -1, ax)).view(rows, lh)
    return h


@torch.no_grad()
def observables_int8(c0: torch.Tensor, c1: torch.Tensor, shape: tuple[int, ...]):
    """``(m, e)`` per spin from the two int8 colour arrays."""
    n = c0.numel() * 2
    m = (c0.sum(dtype=torch.int64) + c1.sum(dtype=torch.int64)).item() / n
    e = -(c0.to(torch.int32) * _neighbor_sum(c1, shape, 0)).sum(dtype=torch.int64).item() / n
    return m, e


def popcount64(x: torch.Tensor) -> torch.Tensor:
    """Per-element popcount of int64 tensors (SWAR), returned as int64."""
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return ((x * 0x0101010101010101) >> 56) & 0xFF


def _shift_right1(o: torch.Tensor) -> torch.Tensor:
    """Logical ``o >> 1`` for int64 bit patterns."""
    return (o >> 1) & 0x7FFFFFFFFFFFFFFF


@torch.no_grad()
def observables_packed(w0: torch.Tensor, w1: torch.Tensor, shape: tuple[int, ...]):
    """``(m, e)`` per spin from packed 2D colour arrays (int64 views of uint64 words)."""
    n = w0.numel() * 128
    up = popcount64(w0).sum() + popcount64(w1).sum()
    m = (2 * up.item() - n) / n
    o = w1
    right = _shift_right1(o) | (torch.roll(o, -1, 1) << 63)
    left = (o << 1) | ((torch.roll(o, 1, 1) >> 63) & 1)
    par = torch.as_tensor(row_parity(shape) % 2 == 1, device=o.device)[:, None]
    horiz = torch.where(par, right, left)
    anti = 0
    for nb in (o, horiz, torch.roll(o, 1, 0), torch.roll(o, -1, 0)):
        anti += popcount64(w0 ^ nb).sum().item()
    bonds = n * 2  # two bonds per spin in 2D
    e = -(bonds - 2 * anti) / n
    return m, e
