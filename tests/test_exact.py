import math

import numpy as np
import pytest

from pyising.exact import (
    TC_2D,
    energy_2d,
    energy_torus,
    log_partition_torus,
    magnetization_2d,
)


def brute_force_4x4(T):
    confs = ((np.arange(2**16)[:, None] >> np.arange(16)) & 1).astype(np.int64) * 2 - 1
    confs = confs.reshape(-1, 4, 4)
    E = -(confs * np.roll(confs, 1, 1)).sum((1, 2)) - (confs * np.roll(confs, 1, 2)).sum((1, 2))
    w = np.exp(-(E - E.min()) / T)
    return np.log(w.sum()) - E.min() / T, (E * w).sum() / w.sum() / 16


@pytest.mark.parametrize("T", [1.0, 2.0, TC_2D, 3.0, 5.0])
def test_kaufman_matches_enumeration(T):
    lnz, e = brute_force_4x4(T)
    assert log_partition_torus(1 / T, 4, 4) == pytest.approx(lnz, rel=1e-12)
    assert energy_torus(T, 4) == pytest.approx(e, abs=1e-8)


def test_onsager_values():
    assert TC_2D == pytest.approx(2.269185314213022, rel=1e-14)
    assert energy_2d(TC_2D) == pytest.approx(-math.sqrt(2), rel=1e-9)
    assert magnetization_2d(3.0) == 0.0
    assert magnetization_2d(2.0) == pytest.approx((1 - math.sinh(1.0) ** -4) ** 0.125)


@pytest.mark.parametrize("T", [1.8, 2.6, 3.5])
def test_torus_approaches_infinite_lattice(T):
    assert energy_torus(T, 256) == pytest.approx(energy_2d(T), abs=1e-6)
