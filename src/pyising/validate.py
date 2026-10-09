"""Physics validation suite -> ``results/validation.json``.

Checks (all numbers land in the JSON, figures and paper read them from there):

1. 2D energy per spin vs the *exact finite-lattice* value (Kaufman / Ferdinand-Fisher) at every
   temperature, for Metropolis (int8 and multi-spin coding) and heat bath.
2. 2D |m| vs the Onsager-Yang spontaneous magnetisation (away from T_c, where finite-size
   effects are small).
3. Binder-cumulant crossings: 2D (Wolff, several L) vs Onsager's T_c; 3D vs the simple-cubic
   critical coupling of Ferrenberg, Xu & Landau (2018).
4. 3D GPU Metropolis / heat bath vs 3D Wolff energies (statistical agreement).
5. Bitwise identity of every backend that shares a random stream.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import backends as B
from .cluster import Wolff
from .exact import BETA_C_3D, TC_2D, TC_3D, energy_torus, magnetization_2d
from .lattice import random_lattice
from .observables import crossing, sample

QUICK = {
    "L2": 128, "T2": [1.8, 2.1, TC_2D, 2.5, 3.0], "therm": 1000, "meas": 500, "every": 4,
    "binder2_L": [8, 16], "binder2_T": list(np.linspace(2.2, 2.34, 5)), "binder_meas": 1000,
    "binder3_L": [4, 8], "binder3_T": list(np.linspace(4.45, 4.57, 5)),
    "L3": 16, "T3": [4.0, TC_3D, 5.0],
}
FULL = {
    "L2": 128, "T2": sorted(list(np.round(np.linspace(1.6, 3.4, 19), 3)) + [TC_2D]),
    "therm": 2000, "meas": 2000, "every": 10,
    "binder2_L": [8, 16, 32, 64], "binder2_T": list(np.round(np.linspace(2.20, 2.34, 15), 4)),
    "binder_meas": 10000,
    "binder3_L": [8, 16, 32], "binder3_T": list(np.round(np.linspace(4.47, 4.55, 9), 4)),
    "L3": 32, "T3": [3.5, 4.2, TC_3D, 5.0, 6.0],
}


def equilibrated(shape: tuple[int, ...], T: float, seed: int, sweeps: int = 300) -> np.ndarray:
    """Equilibrium start from Wolff cluster updates (removes the critical-slowing-down
    thermalisation bias of local updates started hot or cold)."""
    w = Wolff(shape, 1.0 / T, seed=seed)
    w.sweep(sweeps)
    return w.spins.reshape(shape).copy()


def _n_meas(cfg: dict, T: float, tc: float) -> int:
    """Four times more measurements within 0.15 of T_c, where autocorrelations are longest."""
    return cfg["meas"] * (4 if abs(T - tc) < 0.15 * tc / 2.27 else 1)


def _pick(variant: str, dim: int = 2, algorithm: str = "metropolis") -> tuple[str, str]:
    """Fastest available backend for a variant (CUDA C > Triton > ... > NumPy)."""
    for be in ("cupy-cuda-c", "triton", "warp", "numba", "torch-compile", "torch-eager",
               "numpy"):
        if (be, variant) in B.REGISTRY:
            cls = B.load(be, variant)
            if cls is not None and dim in cls.dims and algorithm in cls.algorithms:
                return be, variant
    raise RuntimeError(f"no backend for {variant}")


def check_identity(log=print) -> list[dict]:
    """Same seed + same start -> identical configurations for every backend sharing a stream."""
    out = []
    cases = [((32, 64), "metropolis", "int8"), ((32, 64), "heatbath", "int8"),
             ((8, 8, 16), "metropolis", "int8"), ((8, 8, 16), "heatbath", "int8"),
             ((128, 256), "metropolis", "msc")]
    for shape, alg, stream in cases:
        lat = random_lattice(shape, 2024)
        ref_cls = B.load("numpy", "int8" if stream == "int8" else "msc")
        ref = ref_cls(shape, 0.42 if len(shape) == 2 else 0.22, seed=99, algorithm=alg, init=lat)
        ref.sweep(8)
        ref_lat = ref.get_lattice()
        for be, var in B.REGISTRY:
            cls = B.load(be, var)
            if cls is None or be == "numpy" or cls.stream != stream:
                continue
            if len(shape) not in cls.dims or alg not in cls.algorithms:
                continue
            sim = cls(shape, ref.beta, seed=99, algorithm=alg, init=lat)
            sim.sweep(8)
            same = bool(np.array_equal(sim.get_lattice(), ref_lat))
            out.append({"backend": be, "variant": var, "shape": list(shape), "algorithm": alg,
                        "identical_to_numpy": same})
            log(f"  identity {be:>14}/{var:<18} {alg:<10} {shape}: {same}")
    return out


def energy_scan_2d(cfg: dict, log=print) -> list[dict]:
    """2D |m|(T), e(T) for Metropolis int8, Metropolis msc, heat bath int8."""
    rows = []
    L = cfg["L2"]
    for variant, alg in (("int8", "metropolis"), ("msc", "metropolis"), ("int8", "heatbath")):
        be, var = _pick(variant, 2, alg)
        for T in cfg["T2"]:
            sim = B.create(be, var, shape=(L, L), beta=1.0 / T, seed=7, algorithm=alg,
                           init=equilibrated((L, L), T, seed=int(1000 * T)))
            s = sample(sim, cfg["therm"], _n_meas(cfg, T, TC_2D), cfg["every"]).summary()
            e_ex = energy_torus(T, L)
            row = {"backend": be, "variant": var, "algorithm": alg, "L": L, "T": float(T),
                   **s, "e_exact_L": e_ex, "e_exact_inf": float(_e_inf(T)),
                   "m_exact_inf": float(magnetization_2d(T))}
            row["e_z"] = (s["e"] - e_ex) / s["e_err"] if s["e_err"] > 0 else float("nan")
            rows.append(row)
            log(f"  2D {var:<5} {alg:<10} T={T:.4f} |m|={s['abs_m']:.5f} "
                f"(Yang {row['m_exact_inf']:.5f}) e={s['e']:.6f}+-{s['e_err']:.6f} "
                f"(exact L={L}: {e_ex:.6f}, z={row['e_z']:+.2f})")
    return rows


def _e_inf(T: float) -> float:
    from .exact import energy_2d

    return float(energy_2d(T))


def binder_scan(dim: int, Ls: list[int], Ts: list[float], n_meas: int, log=print) -> dict:
    """Wolff Binder cumulants and pairwise crossings."""
    data: dict = {"dim": dim, "L": Ls, "T": [float(t) for t in Ts], "U": {}, "U_err": {}}
    for L in Ls:
        us, errs = [], []
        t0 = time.time()
        for T in Ts:
            w = Wolff((L,) * dim, 1.0 / T, seed=1000 * L + int(T * 1000))
            s = sample(w, 200, n_meas, 1).summary()
            us.append(s["binder"])
            errs.append(s["binder_err"])
        data["U"][str(L)] = us
        data["U_err"][str(L)] = errs
        log(f"  Binder {dim}D L={L}: " + " ".join(f"{u:.4f}" for u in us)
            + f"  ({time.time() - t0:.0f} s)")
    data["crossings"] = []
    for a, b in zip(Ls[:-1], Ls[1:], strict=True):
        tc = crossing(np.array(Ts), np.array(data["U"][str(a)]), np.array(data["U"][str(b)]))
        data["crossings"].append({"L1": a, "L2": b, "Tc": tc})
        log(f"  crossing {dim}D L={a}/{b}: Tc = {tc:.5f}")
    ref = TC_2D if dim == 2 else TC_3D
    data["Tc_reference"] = ref
    data["Tc_estimate"] = data["crossings"][-1]["Tc"]
    data["Tc_rel_error"] = abs(data["Tc_estimate"] - ref) / ref
    return data


def scan_3d(cfg: dict, log=print) -> list[dict]:
    """3D Metropolis and heat bath (GPU) vs Wolff (CPU) energies at the same L."""
    rows = []
    L = cfg["L3"]
    for T in cfg["T3"]:
        w = Wolff((L, L, L), 1.0 / T, seed=5)
        ws = sample(w, 200, _n_meas(cfg, T, TC_3D), 1).summary()
        for alg in ("metropolis", "heatbath"):
            be, var = _pick("int8", 3, alg)
            sim = B.create(be, var, shape=(L, L, L), beta=1.0 / T, seed=11, algorithm=alg,
                           init=w.spins.reshape(L, L, L).copy())
            s = sample(sim, cfg["therm"], _n_meas(cfg, T, TC_3D), cfg["every"]).summary()
            err = float(np.hypot(s["e_err"], ws["e_err"]))
            rows.append({"backend": be, "variant": var, "algorithm": alg, "L": L, "T": float(T),
                         **s, "wolff_e": ws["e"], "wolff_e_err": ws["e_err"],
                         "wolff_abs_m": ws["abs_m"], "e_z": (s["e"] - ws["e"]) / err})
            log(f"  3D {alg:<10} T={T:.4f} e={s['e']:.5f}+-{s['e_err']:.5f} "
                f"Wolff {ws['e']:.5f}+-{ws['e_err']:.5f}")
    return rows


def run(quick: bool = False, out: str | Path | None = "results/validation.json", log=print):
    """Run the whole suite; returns the result dict and writes it as JSON."""
    cfg = QUICK if quick else FULL
    t0 = time.time()
    res: dict = {"quick": quick, "config": {k: v for k, v in cfg.items()}}
    log("[1/5] bitwise identity across backends")
    res["identity"] = check_identity(log)
    log("[2/5] 2D energy and magnetisation vs exact results")
    res["scan2d"] = energy_scan_2d(cfg, log)
    log("[3/5] 2D Binder crossing (Wolff)")
    res["binder2d"] = binder_scan(2, cfg["binder2_L"], cfg["binder2_T"], cfg["binder_meas"], log)
    log("[4/5] 3D Binder crossing (Wolff)")
    res["binder3d"] = binder_scan(3, cfg["binder3_L"], cfg["binder3_T"], cfg["binder_meas"], log)
    res["binder3d"]["beta_c_reference"] = BETA_C_3D
    log("[5/5] 3D GPU checkerboard vs Wolff")
    res["scan3d"] = scan_3d(cfg, log)
    res["summary"] = summarize(res)
    res["seconds"] = time.time() - t0
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1, default=float))
        log(f"wrote {out} ({res['seconds']:.0f} s)")
    return res


def summarize(res: dict) -> dict:
    """Headline validation numbers."""
    s2 = res["scan2d"]
    away = [r for r in s2 if abs(r["T"] - TC_2D) > 0.2 and r["T"] < TC_2D]
    return {
        "all_identical": all(r["identical_to_numpy"] for r in res["identity"]),
        "n_identity_checks": len(res["identity"]),
        "e_max_abs_err_vs_exact_L": max(abs(r["e"] - r["e_exact_L"]) for r in s2),
        "e_max_abs_z": max(abs(r["e_z"]) for r in s2),
        "m_max_abs_err_vs_yang_away_from_Tc": max(
            (abs(r["abs_m"] - r["m_exact_inf"]) for r in away), default=float("nan")),
        "Tc2d_estimate": res["binder2d"]["Tc_estimate"],
        "Tc2d_rel_error": res["binder2d"]["Tc_rel_error"],
        "Tc3d_estimate": res["binder3d"]["Tc_estimate"],
        "Tc3d_rel_error": res["binder3d"]["Tc_rel_error"],
        "e3d_max_abs_z_vs_wolff": max(abs(r["e_z"]) for r in res["scan3d"]),
    }
