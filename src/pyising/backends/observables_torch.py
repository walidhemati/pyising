"""On-device magnetisation and energy from compressed (int8) or packed (uint64) colour arrays.

Every bond has exactly one colour-0 end, so ``E = -sum_{i in colour 0} s_i h_i``. Reductions
run over blocks of rows, so the extra memory is bounded (~64 MB) even for multi-GB lattices.
"""

from __future__ import annotations

import torch

from ..lattice import row_parity

CHUNK_ELEMS = 1 << 24


def device_random_int8(n: int, seed: int) -> torch.Tensor:
    """``n`` i.i.d. +-1 int8 spins generated on the GPU in place (no host round trip)."""
    g = torch.Generator(device="cuda").manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
    return torch.randint(0, 2, (n,), generator=g, device="cuda", dtype=torch.int8).mul_(2).sub_(1)


def device_random_words(n: int, seed: int) -> torch.Tensor:
    """``n`` uniformly random 64-bit words (int64 tensor) generated on the GPU."""
    g = torch.Generator(device="cuda").manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
    return torch.randint(-2**63, 2**63 - 1, (n,), generator=g, device="cuda", dtype=torch.int64)


def _row_neighbors(rows: torch.Tensor, shape: tuple[int, ...]) -> list[torch.Tensor]:
    """Flattened row indices of the +-1 neighbours along every non-x axis."""
    ly = shape[-2]
    y = rows % ly
    out = [rows - y + (y + 1) % ly, rows - y + (y - 1) % ly]
    if len(shape) == 3:
        lz = shape[0]
        z = rows // ly
        out += [((z + 1) % lz) * ly + y, ((z - 1) % lz) * ly + y]
    return out


def _chunks(n_rows: int, row_len: int):
    step = max(1, CHUNK_ELEMS // max(1, row_len))
    for a in range(0, n_rows, step):
        yield a, min(n_rows, a + step)


@torch.no_grad()
def observables_int8(c0: torch.Tensor, c1: torch.Tensor, shape: tuple[int, ...]):
    """``(m, e)`` per spin from the two int8 colour arrays (colour-0 sites own the bonds)."""
    rows, lh = c0.shape
    n = 2 * rows * lh
    par_all = torch.as_tensor(row_parity(shape) % 2 == 1, device=c0.device)
    msum = 0
    esum = 0
    for a, b in _chunks(rows, lh):
        idx = torch.arange(a, b, device=c0.device)
        s = c0[a:b].to(torch.int32)
        o = c1[a:b].to(torch.int32)
        msum += int(s.sum()) + int(o.sum())
        h = o + torch.where(par_all[a:b, None], torch.roll(o, -1, 1), torch.roll(o, 1, 1))
        for nb in _row_neighbors(idx, shape):
            h += c1.index_select(0, nb).to(torch.int32)
        esum += int((s * h).sum(dtype=torch.int64))
    return msum / n, -esum / n


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
    rows, nw = w0.shape
    n = rows * nw * 128
    par_all = torch.as_tensor(row_parity(shape) % 2 == 1, device=w0.device)
    up = 0
    anti = 0
    for a, b in _chunks(rows, nw * 8):
        idx = torch.arange(a, b, device=w0.device)
        s = w0[a:b]
        o = w1[a:b]
        up += int(popcount64(s).sum()) + int(popcount64(o).sum())
        right = _shift_right1(o) | (torch.roll(o, -1, 1) << 63)
        left = (o << 1) | ((torch.roll(o, 1, 1) >> 63) & 1)
        horiz = torch.where(par_all[a:b, None], right, left)
        anti += int(popcount64(s ^ o).sum()) + int(popcount64(s ^ horiz).sum())
        for nb in _row_neighbors(idx, shape):
            anti += int(popcount64(s ^ w1.index_select(0, nb)).sum())
    m = (2 * up - n) / n
    bonds = n * len(shape)
    return m, -(bonds - 2 * anti) / n
