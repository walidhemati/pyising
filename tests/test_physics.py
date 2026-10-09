"""Statistical physics checks on CPU (NumPy reference + Wolff)."""

import numpy as np
import pytest

from pyising.backends.numpy_backend import NumpyCheckerboard, NumpyMultiSpin
from pyising.cluster import Wolff
from pyising.exact import TC_2D, energy_torus
from pyising.observables import binder, crossing, sample


@pytest.mark.parametrize("cls,alg", [(NumpyCheckerboard, "metropolis"),
                                     (NumpyCheckerboard, "heatbath"),
                                     (NumpyMultiSpin, "metropolis")])
@pytest.mark.parametrize("T", [1.8, 3.0])
def test_energy_matches_exact_torus(cls, alg, T):
    L = 16 if cls is NumpyCheckerboard else 128
    sim = cls((L, L), 1 / T, seed=3, algorithm=alg, init="cold")
    n_meas = 1500 if L == 16 else 400
    s = sample(sim, 200, n_meas, 2).summary()
    assert abs(s["e"] - energy_torus(T, L)) < 5 * s["e_err"] + 2e-3


def test_wolff_energy_and_binder():
    T = 2.4
    w = Wolff((16, 16), 1 / T, seed=2)
    s = sample(w, 100, 4000, 1).summary()
    assert abs(s["e"] - energy_torus(T, 16)) < 5 * s["e_err"] + 1e-3
    assert 0.0 < s["binder"] < 2 / 3


def test_binder_limits_and_crossing():
    rng = np.random.default_rng(0)
    assert binder(rng.normal(size=200000)) == pytest.approx(0.0, abs=0.02)
    assert binder(np.ones(10)) == pytest.approx(2 / 3)
    Ts = np.array([2.0, 2.2, 2.4])
    assert crossing(Ts, np.array([0.6, 0.5, 0.4]), np.array([0.65, 0.5, 0.3])) == \
        pytest.approx(2.2)


def test_low_temperature_ordered():
    sim = NumpyCheckerboard((32, 32), 1 / 1.0, seed=1, init="cold")
    sim.sweep(50)
    assert abs(sim.magnetization()) > 0.99
    assert TC_2D > 2.26
