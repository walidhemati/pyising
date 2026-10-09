"""Measurement helpers: time series, binning errors, Binder cumulant."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Series:
    """Time series of magnetisation ``m`` and energy ``e`` per spin."""

    m: np.ndarray
    e: np.ndarray

    def summary(self) -> dict:
        am = np.abs(self.m)
        m2, m4 = self.m**2, self.m**4
        return {
            "abs_m": float(am.mean()), "abs_m_err": binning_error(am),
            "e": float(self.e.mean()), "e_err": binning_error(self.e),
            "m2": float(m2.mean()), "m4": float(m4.mean()),
            "binder": binder(self.m), "binder_err": jackknife_binder_error(self.m),
            "n": int(self.m.size),
        }


def binning_error(x: np.ndarray, n_bins: int = 32) -> float:
    """Standard error of the mean from ``n_bins`` equal blocks (robust to autocorrelation when
    blocks are longer than the integrated autocorrelation time)."""
    x = np.asarray(x, dtype=float)
    nb = min(n_bins, x.size)
    if nb < 2:
        return float("nan")
    blocks = x[: x.size // nb * nb].reshape(nb, -1).mean(axis=1)
    return float(blocks.std(ddof=1) / np.sqrt(nb))


def binder(m: np.ndarray) -> float:
    """Binder cumulant ``U_4 = 1 - <m^4> / (3 <m^2>^2)``."""
    m = np.asarray(m, dtype=float)
    return float(1.0 - np.mean(m**4) / (3.0 * np.mean(m**2) ** 2))


def jackknife_binder_error(m: np.ndarray, n_bins: int = 32) -> float:
    """Blocked jackknife error of :func:`binder`."""
    m = np.asarray(m, dtype=float)
    nb = min(n_bins, m.size)
    if nb < 2:
        return float("nan")
    blocks = m[: m.size // nb * nb].reshape(nb, -1)
    m2 = (blocks**2).mean(axis=1)
    m4 = (blocks**4).mean(axis=1)
    s2, s4 = m2.sum(), m4.sum()
    jk = np.array([1 - ((s4 - m4[i]) / (nb - 1)) / (3 * ((s2 - m2[i]) / (nb - 1)) ** 2)
                   for i in range(nb)])
    return float(np.sqrt((nb - 1) * np.mean((jk - jk.mean()) ** 2)))


def sample(sim, n_therm: int, n_meas: int, every: int = 1) -> Series:
    """Thermalise ``sim`` for ``n_therm`` sweeps, then record ``n_meas`` measurements spaced by
    ``every`` sweeps. Works for any backend and for :class:`pyising.cluster.Wolff`."""
    step = sim.sweep
    step(n_therm)
    m = np.empty(n_meas)
    e = np.empty(n_meas)
    for i in range(n_meas):
        step(every)
        if hasattr(sim, "observables"):
            m[i], e[i] = sim.observables()
        else:
            m[i], e[i] = sim.magnetization(), sim.energy()
    return Series(m, e)


def crossing(Ts: np.ndarray, u_a: np.ndarray, u_b: np.ndarray) -> float:
    """Temperature where two Binder curves cross (linear interpolation of their difference)."""
    d = np.asarray(u_a) - np.asarray(u_b)
    for i in range(len(d) - 1):
        if d[i] == 0:
            return float(Ts[i])
        if d[i] * d[i + 1] < 0:
            return float(Ts[i] - d[i] * (Ts[i + 1] - Ts[i]) / (d[i + 1] - d[i]))
    return float("nan")
