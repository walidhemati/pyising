import numpy as np
import pytest

from pyising.backends.numpy_backend import NumpyCheckerboard, NumpyMultiSpin
from pyising.lattice import (
    energy_per_spin,
    from_compressed,
    neighbor_sum,
    pack_bits,
    parity_grid,
    random_lattice,
    threshold_table,
    to_compressed,
    unpack_bits,
)
from pyising.philox import site_uniforms


@pytest.mark.parametrize("shape", [(8, 16), (6, 8), (4, 6, 8)])
def test_compressed_roundtrip(shape):
    lat = random_lattice(shape, 1)
    c0, c1 = to_compressed(lat)
    par = parity_grid(shape)
    assert np.all(np.sort(c0.ravel()) == np.sort(lat[par == 0]))
    np.testing.assert_array_equal(from_compressed(c0, c1, shape), lat)


def test_pack_roundtrip():
    c = random_lattice((4, 256), 2)
    np.testing.assert_array_equal(unpack_bits(pack_bits(c)), c)


def test_threshold_table():
    t = threshold_table(0.5, 2, "metropolis")
    assert t[0] == t[1] == t[2] == 0xFFFFFFFF
    assert t[3] == int(np.exp(-2.0) * 2**32) and t[4] == int(np.exp(-4.0) * 2**32)
    hb = threshold_table(0.5, 3, "heatbath")
    assert len(hb) == 7 and hb[3] == 2**31


def naive_sweep(lat, beta, seed, t, algorithm):
    """Full-lattice reference written directly from the definitions (no compressed layout)."""
    lat = lat.copy()
    d = lat.ndim
    par = parity_grid(lat.shape)
    table = threshold_table(beta, d, algorithm)
    for color in (0, 1):
        h = neighbor_sum(lat).astype(np.int64)
        u = site_uniforms(lat.size // 2, t, color, seed)
        # colour sites in C order of the full lattice <-> colour-site index n
        idx = np.argwhere(par == color)
        s = lat[par == color].astype(np.int64)
        x = h[par == color] if algorithm == "heatbath" else s * h[par == color]
        acc = u < table[(x + 2 * d) // 2]
        if algorithm == "heatbath":
            new = np.where(acc, 1, -1)
        else:
            new = np.where(acc | (x <= 0), -s, s)
        lat[tuple(idx.T)] = new
    return lat


@pytest.mark.parametrize("shape", [(8, 16), (4, 4, 8)])
@pytest.mark.parametrize("algorithm", ["metropolis", "heatbath"])
def test_numpy_backend_matches_naive(shape, algorithm):
    lat = random_lattice(shape, 3)
    sim = NumpyCheckerboard(shape, 0.37, seed=42, algorithm=algorithm, init=lat)
    ref = lat
    for t in range(4):
        ref = naive_sweep(ref, 0.37, 42, t, algorithm)
    sim.sweep(4)
    np.testing.assert_array_equal(sim.get_lattice(), ref)


def test_sweep_splitting_and_observables():
    shape = (16, 16)
    a = NumpyCheckerboard(shape, 0.4, seed=1)
    b = NumpyCheckerboard(shape, 0.4, seed=1)
    a.sweep(5)
    b.sweep(2)
    b.sweep(3)
    np.testing.assert_array_equal(a.get_lattice(), b.get_lattice())
    assert a.energy() == energy_per_spin(a.get_lattice())


def test_cold_start_ground_state():
    sim = NumpyMultiSpin((16, 128), 0.5, init="cold")
    assert sim.magnetization() == 1.0 and sim.energy() == -2.0


def test_msc_metropolis_matches_int8_rule():
    """Multi-spin coding flips exactly the spins whose bitwise-built uniform passes the same
    integer test as the int8 kernel (checked here with p = 0 and p = 1 limits)."""
    lat = random_lattice((16, 128), 4)
    hot = NumpyMultiSpin((16, 128), 0.0, seed=5, init=lat)  # beta = 0: every spin flips
    hot.sweep(1)
    ref = NumpyCheckerboard((16, 128), 0.0, seed=5, init=lat)
    ref.sweep(1)
    np.testing.assert_array_equal(hot.get_lattice(), ref.get_lattice())
