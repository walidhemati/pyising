"""Exact results for the square-lattice Ising model (J = k_B = 1) and 3D reference values."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ellipk

#: Onsager (1944): ``sinh(2 / T_c) = 1``.
TC_2D = 2.0 / math.log(1.0 + math.sqrt(2.0))
BETA_C_2D = 1.0 / TC_2D
#: Simple-cubic critical coupling, Ferrenberg, Xu & Landau, Phys. Rev. E 97, 043301 (2018).
BETA_C_3D = 0.221654626
BETA_C_3D_ERR = 0.000000005
TC_3D = 1.0 / BETA_C_3D


def magnetization_2d(T: np.ndarray | float) -> np.ndarray:
    """Spontaneous magnetisation (Onsager/Yang 1952): ``(1 - sinh(2/T)^-4)^(1/8)`` below T_c."""
    T = np.asarray(T, dtype=float)
    s = np.sinh(2.0 / T)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(T < TC_2D, np.clip(1.0 - s**-4, 0.0, None) ** 0.125, 0.0)
    return m


def energy_2d(T: np.ndarray | float) -> np.ndarray:
    """Onsager's internal energy per spin of the infinite square lattice.

    ``u = -coth(2K) [1 + (2/pi) (2 tanh^2(2K) - 1) K(k^2)]`` with ``k = 2 sinh 2K / cosh^2 2K``.
    """
    K = 1.0 / np.asarray(T, dtype=float)
    k = 2.0 * np.sinh(2 * K) / np.cosh(2 * K) ** 2
    m = np.minimum(k * k, 1.0 - 1e-16)
    return -1.0 / np.tanh(2 * K) * (1.0 + 2.0 / np.pi * (2.0 * np.tanh(2 * K) ** 2 - 1.0) * ellipk(m))


def _log_2cosh(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    return ax + np.log1p(np.exp(-2.0 * ax))


def _log_abs_2sinh(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ax = np.abs(x)
    with np.errstate(divide="ignore"):
        return ax + np.log1p(-np.exp(-2.0 * ax)), np.sign(x)


def log_partition_torus(K: float, m: int, n: int) -> float:
    """``ln Z`` of the ``m x n`` periodic square lattice (Kaufman 1949, Ferdinand & Fisher 1969).

    ``Z = 1/2 (2 sinh 2K)^(mn/2) (Z1 + Z2 + Z3 + Z4)`` with ``Z1,2 = prod_r 2cosh / 2sinh
    (m g_{2r+1} / 2)``, ``Z3,4`` the same with ``g_{2r}``, ``cosh g_l = cosh 2K coth 2K -
    cos(pi l / n)`` and ``g_0 = 2K + ln tanh K`` (which changes sign at T_c).
    """
    r = np.arange(n)
    c = math.cosh(2 * K) / math.tanh(2 * K)

    def gamma(idx: np.ndarray) -> np.ndarray:
        g = np.arccosh(np.maximum(c - np.cos(np.pi * idx / n), 1.0))
        return np.where(idx == 0, 2 * K + math.log(math.tanh(K)), g)

    g_odd, g_even = gamma(2 * r + 1), gamma(2 * r)
    terms = []
    for g in (g_odd, g_even):
        terms.append((np.sum(_log_2cosh(m * g / 2)), 1.0))
        ls, sg = _log_abs_2sinh(m * g / 2)
        terms.append((np.sum(ls), float(np.prod(sg))))
    lmax = max(t[0] for t in terms)
    total = sum(sg * math.exp(lv - lmax) for lv, sg in terms if np.isfinite(lv))
    return (-math.log(2.0) + 0.5 * m * n * math.log(2 * math.sinh(2 * K)) + lmax
            + math.log(total))


def energy_torus(T: float, L: int, h: float = 1e-5) -> float:
    """Exact energy per spin of the finite ``L x L`` torus, ``-d ln Z / dK / L^2``."""
    K = 1.0 / T
    d = (log_partition_torus(K + h, L, L) - log_partition_torus(K - h, L, L)) / (2 * h)
    return -d / (L * L)


def specific_heat_torus(T: float, L: int, h: float = 1e-4) -> float:
    """Exact specific heat per spin of the finite torus, ``K^2 d^2 ln Z / dK^2 / L^2``."""
    K = 1.0 / T
    f = [log_partition_torus(K + s * h, L, L) for s in (-1, 0, 1)]
    return K * K * (f[0] - 2 * f[1] + f[2]) / (h * h) / (L * L)
