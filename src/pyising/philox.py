"""Philox4x32-10 counter-based random number generator (Salmon et al., SC'11), NumPy version.

Every backend in :mod:`pyising.backends` implements the same function, so the random number that
decides the fate of a spin depends only on (site, sweep, colour, seed) and never on the thread
layout. The NumPy version below is the oracle the GPU kernels are tested against.
"""

from __future__ import annotations

import numpy as np

M0 = 0xD2511F53
M1 = 0xCD9E8D57
W0 = 0x9E3779B9
W1 = 0xBB67AE85
MASK32 = 0xFFFFFFFF
ROUNDS = 10


def split_seed(seed: int) -> tuple[int, int]:
    """Split a 64-bit seed into the two 32-bit Philox key words (lo, hi)."""
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    return seed & MASK32, seed >> 32


def philox4x32(
    c0: np.ndarray | int,
    c1: np.ndarray | int,
    c2: np.ndarray | int,
    c3: np.ndarray | int,
    k0: int,
    k1: int,
    rounds: int = ROUNDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised Philox4x32-R. Counters broadcast; returns four uint32 arrays."""
    u64 = np.uint64
    c0, c1, c2, c3 = (np.asarray(c, dtype=u64) & u64(MASK32) for c in (c0, c1, c2, c3))
    c0, c1, c2, c3 = np.broadcast_arrays(c0, c1, c2, c3)
    k0, k1 = int(k0) & MASK32, int(k1) & MASK32
    m0, m1, mask = u64(M0), u64(M1), u64(MASK32)
    s32 = u64(32)
    for r in range(rounds):
        p0 = c0 * m0
        p1 = c2 * m1
        hi0, lo0 = p0 >> s32, p0 & mask
        hi1, lo1 = p1 >> s32, p1 & mask
        c0, c1, c2, c3 = hi1 ^ c1 ^ u64(k0), lo1, hi0 ^ c3 ^ u64(k1), lo0
        if r != rounds - 1:
            k0 = (k0 + W0) & MASK32
            k1 = (k1 + W1) & MASK32
    return tuple(c.astype(np.uint32) for c in (c0, c1, c2, c3))  # type: ignore[return-value]


def site_uniforms(n_sites: int, sweep: int, color: int, seed: int) -> np.ndarray:
    """uint32 random word for colour-sites 0..n_sites-1 of one half-sweep (int8 stream).

    Site ``n`` takes word ``n % 4`` of ``Philox(ctr=(n // 4, sweep, color, 0), key=seed)``.
    """
    if n_sites % 4:
        raise ValueError("number of colour sites must be a multiple of 4")
    k0, k1 = split_seed(seed)
    q = np.arange(n_sites // 4, dtype=np.uint64)
    r = philox4x32(q, sweep, color, 0, k0, k1)
    return np.stack(r, axis=1).reshape(-1)


def word_randoms(n_words: int, sweep: int, color: int, seed: int, level_pair: int) -> tuple:
    """Two uint64 random words per packed spin word (multi-spin-coding stream).

    Word ``w`` at bit-level ``2*level_pair + i`` uses ``Philox(ctr=(w, sweep, color,
    1 + level_pair))``; level ``2l`` is ``r0 | r1 << 32`` and level ``2l + 1`` is ``r2 | r3 << 32``.
    """
    k0, k1 = split_seed(seed)
    w = np.arange(n_words, dtype=np.uint64)
    r0, r1, r2, r3 = (x.astype(np.uint64) for x in philox4x32(w, sweep, color, 1 + level_pair,
                                                                k0, k1))
    s = np.uint64(32)
    return r0 | (r1 << s), r2 | (r3 << s)
