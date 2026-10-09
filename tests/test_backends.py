"""Every GPU backend reproduces the NumPy oracle bit for bit (same Philox stream)."""

import numpy as np
import pytest

from pyising import backends as B
from pyising.lattice import energy_per_spin, magnetization, random_lattice

pytestmark = pytest.mark.gpu

INT8 = [k for k, v in B.REGISTRY.items() if k[0] != "numpy" and "int8" in k[1]]
MSC = [k for k, v in B.REGISTRY.items() if k[0] != "numpy" and "msc" in k[1]]


def _cls(key):
    cls = B.load(*key)
    if cls is None:
        pytest.skip(f"{key} unavailable")
    return cls


@pytest.mark.parametrize("key", INT8, ids=lambda k: "/".join(k))
@pytest.mark.parametrize("shape", [(16, 24), (8, 8, 16), (64, 32)])
@pytest.mark.parametrize("algorithm", ["metropolis", "heatbath"])
def test_int8_bitwise_identical(key, shape, algorithm):
    cls = _cls(key)
    lat = random_lattice(shape, 5)
    beta = 0.42 if len(shape) == 2 else 0.22
    ref = B.load("numpy", "int8")(shape, beta, seed=2**40 + 7, algorithm=algorithm, init=lat)
    sim = cls(shape, beta, seed=2**40 + 7, algorithm=algorithm, init=lat)
    ref.sweep(6)
    sim.sweep(6)
    np.testing.assert_array_equal(sim.get_lattice(), ref.get_lattice())
    m, e = sim.observables()
    assert m == pytest.approx(magnetization(ref.get_lattice()))
    assert e == pytest.approx(energy_per_spin(ref.get_lattice()))


@pytest.mark.parametrize("key", MSC, ids=lambda k: "/".join(k))
@pytest.mark.parametrize("shape", [(128, 128), (64, 384)])
def test_msc_bitwise_identical(key, shape):
    cls = _cls(key)
    lat = random_lattice(shape, 9)
    ref = B.load("numpy", "msc")(shape, 0.44, seed=11, init=lat)
    sim = cls(shape, 0.44, seed=11, init=lat)
    ref.sweep(5)
    sim.sweep(5)
    np.testing.assert_array_equal(sim.get_lattice(), ref.get_lattice())
    m, e = sim.observables()
    assert m == pytest.approx(ref.magnetization())
    assert e == pytest.approx(ref.energy())


@pytest.mark.parametrize("key", INT8[:1] + MSC[:1])
def test_set_beta_and_split_sweeps(key):
    cls = _cls(key)
    shape = (128, 128)
    lat = random_lattice(shape, 1)
    a = cls(shape, 0.3, seed=3, init=lat)
    b = cls(shape, 0.3, seed=3, init=lat)
    a.sweep(2)
    a.set_beta(0.5)
    a.sweep(3)
    b.sweep(2)
    b.set_beta(0.5)
    b.sweep(1)
    b.sweep(2)
    np.testing.assert_array_equal(a.get_lattice(), b.get_lattice())


@pytest.mark.parametrize("key", [k for k in B.REGISTRY if k[0] != "numpy"],
                         ids=lambda k: "/".join(k))
def test_hot_and_cold_starts(key):
    cls = _cls(key)
    shape = (128, 128)
    cold = cls(shape, 0.5, init="cold")
    assert cold.observables() == pytest.approx((1.0, -2.0))
    hot = cls(shape, 0.5, init="hot")
    m, e = hot.observables()
    assert abs(m) < 0.05 and abs(e) < 0.05


def test_triton_int64_index_path():
    cls = _cls(("triton", "int8"))

    class Wide(cls):
        force_idx64 = True

    shape = (32, 64)
    lat = random_lattice(shape, 2)
    ref = B.load("numpy", "int8")(shape, 0.4, seed=4, init=lat)
    sim = Wide(shape, 0.4, seed=4, init=lat)
    ref.sweep(3)
    sim.sweep(3)
    np.testing.assert_array_equal(sim.get_lattice(), ref.get_lattice())
