import numpy as np
import pytest

from pyising.philox import philox4x32, site_uniforms, split_seed, word_randoms

# Known-answer vectors from the Random123 distribution (tests/kat_vectors).
KAT = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]


@pytest.mark.parametrize("ctr,key,expected", KAT)
def test_known_answers(ctr, key, expected):
    out = philox4x32(*ctr, *key)
    assert tuple(int(x) for x in out) == expected


def test_vectorised_matches_scalar():
    q = np.arange(10)
    vec = philox4x32(q, 3, 1, 0, 123, 456)
    for i in range(10):
        sc = philox4x32(i, 3, 1, 0, 123, 456)
        assert all(int(v[i]) == int(s) for v, s in zip(vec, sc, strict=True))


def test_site_uniforms_layout():
    u = site_uniforms(16, sweep=5, color=1, seed=2**40 + 3)
    k0, k1 = split_seed(2**40 + 3)
    assert (k0, k1) == (3, 256)
    r = philox4x32(2, 5, 1, 0, k0, k1)
    assert [int(x) for x in u[8:12]] == [int(x) for x in r]


def test_uniformity():
    u = site_uniforms(1 << 16, 0, 0, 1).astype(np.float64) / 2**32
    assert abs(u.mean() - 0.5) < 0.01
    hist, _ = np.histogram(u, bins=16, range=(0, 1))
    chi2 = ((hist - hist.mean()) ** 2 / hist.mean()).sum()
    assert chi2 < 50  # 15 dof


def test_word_randoms_distinct_levels():
    a0, a1 = word_randoms(8, 0, 0, 1, 0)
    b0, _ = word_randoms(8, 0, 0, 1, 1)
    assert a0.dtype == np.uint64
    assert not np.array_equal(a0, a1) and not np.array_equal(a0, b0)


def test_torch_philox_matches_numpy():
    torch = pytest.importorskip("torch")
    from pyising.backends.torch_backend import philox_torch

    q = torch.arange(64, dtype=torch.int64)
    z = torch.zeros_like(q)
    out = philox_torch(q, z + 7, z + 1, z, 0xDEADBEEF, 0x12345678)
    ref = philox4x32(np.arange(64), 7, 1, 0, 0xDEADBEEF, 0x12345678)
    for a, b in zip(out, ref, strict=True):
        np.testing.assert_array_equal(a.numpy().astype(np.uint32), b)
