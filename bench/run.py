"""The only benchmark entry point.

    python bench/run.py --quick   # < 2 min smoke run -> results/dev-quick.json (provisional)
    python bench/run.py --full    # publishable run   -> results/rtx3070-laptop.json

Metric: attempted spin updates per nanosecond ("flips/ns"), at the critical temperature, hot
start, Metropolis unless stated. Minimal DRAM traffic model per update: read own spin, read the
opposite-colour neighbours once, write own spin = 3 spin-sized transfers, i.e. 3 B for int8 and
3/8 B for 1-bit packed spins. Effective bandwidth = flips/ns x bytes/update.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import roofline  # noqa: E402

from pyising import backends as B  # noqa: E402
from pyising.exact import BETA_C_2D, BETA_C_3D  # noqa: E402

FULL_2D = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
FULL_MSC = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
FULL_3D = [64, 128, 256, 512]
PLAN_FULL: list[tuple[str, str, int, list[int], str]] = [
    # backend, variant, dim, sizes, algorithm
    ("numpy", "int8", 2, [1024], "metropolis"),
    ("numpy", "msc", 2, [1024], "metropolis"),
    ("torch-eager", "int8", 2, [1024, 4096, 16384], "metropolis"),
    ("torch-compile", "int8", 2, [1024, 4096, 16384], "metropolis"),
    ("triton", "int8", 2, FULL_2D, "metropolis"),
    ("warp", "int8", 2, FULL_2D, "metropolis"),
    ("numba", "int8", 2, FULL_2D, "metropolis"),
    ("cupy-cuda-c", "int8", 2, FULL_2D + ["max"], "metropolis"),
    ("cupy-cuda-c", "int8-rng-per-site", 2, [4096, 16384], "metropolis"),
    ("triton", "msc", 2, FULL_MSC, "metropolis"),
    ("warp", "msc", 2, FULL_MSC, "metropolis"),
    ("numba", "msc", 2, FULL_MSC, "metropolis"),
    ("cupy-cuda-c", "msc", 2, FULL_MSC + ["max"], "metropolis"),
    ("cupy-cuda-c", "msc-eager32", 2, [4096, 16384, 65536], "metropolis"),
    ("triton", "int8", 2, [16384], "heatbath"),
    ("cupy-cuda-c", "int8", 2, [16384], "heatbath"),
    ("torch-compile", "int8", 3, [64, 256], "metropolis"),
    ("triton", "int8", 3, FULL_3D + [1024], "metropolis"),
    ("warp", "int8", 3, FULL_3D, "metropolis"),
    ("numba", "int8", 3, FULL_3D, "metropolis"),
    ("cupy-cuda-c", "int8", 3, FULL_3D + [1024], "metropolis"),
    ("cupy-cuda-c", "int8", 3, [256], "heatbath"),
]
PLAN_QUICK: list[tuple[str, str, int, list[int], str]] = [
    ("numpy", "int8", 2, [256], "metropolis"),
    ("torch-eager", "int8", 2, [1024], "metropolis"),
    ("torch-compile", "int8", 2, [1024], "metropolis"),
    ("triton", "int8", 2, [1024, 4096], "metropolis"),
    ("warp", "int8", 2, [1024, 4096], "metropolis"),
    ("numba", "int8", 2, [1024, 4096], "metropolis"),
    ("cupy-cuda-c", "int8", 2, [1024, 4096], "metropolis"),
    ("triton", "msc", 2, [1024, 8192], "metropolis"),
    ("warp", "msc", 2, [1024, 8192], "metropolis"),
    ("numba", "msc", 2, [1024, 8192], "metropolis"),
    ("cupy-cuda-c", "msc", 2, [1024, 8192], "metropolis"),
    ("cupy-cuda-c", "msc-eager32", 2, [8192], "metropolis"),
    ("cupy-cuda-c", "int8", 3, [128], "metropolis"),
    ("triton", "int8", 3, [128], "metropolis"),
]


def machine_info() -> dict:
    import torch

    info = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "python": platform.python_version(), "torch": torch.__version__,
            "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    for mod in ("triton", "warp", "cupy", "numba", "numpy"):
        try:
            if mod == "cupy":
                from pyising.backends._gpu import prepare_cupy

                prepare_cupy()
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    try:
        import pynvml

        pynvml.nvmlInit()
        info["driver"] = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(info["driver"], bytes):
            info["driver"] = info["driver"].decode()
    except Exception:  # noqa: BLE001
        info["driver"] = None
    return info


def max_size(cls, dim: int, frac: float = 0.8) -> int:
    """Largest lattice of the variant that fits in ``frac`` of the free device memory."""
    import torch

    free, _ = torch.cuda.mem_get_info()
    bits_per_spin = 8.0 * cls.bytes_per_update / 3.0
    n = frac * free * 8 / bits_per_spin
    L = int(n ** (1.0 / dim))
    m = cls.layout_multiple
    return max(m, (L // m) * m)


def time_one(cls, shape, algorithm: str, beta: float, repeats: int, min_time: float) -> dict:
    """Warm up (incl. JIT), size the run to >= ``min_time`` s, return median/min/max flips/ns."""
    sim = cls(shape, beta, seed=12345, algorithm=algorithm, init="hot")
    sim.sweep(2)
    sim.synchronize()
    t0 = time.perf_counter()
    sim.sweep(1)
    sim.synchronize()
    t1 = time.perf_counter() - t0
    n = max(1, math.ceil(min_time / max(t1, 1e-6)))
    sim.sweep(min(n, 5))  # extra warm-up at the timed launch rate
    sim.synchronize()
    rates = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sim.sweep(n)
        sim.synchronize()
        rates.append(n * sim.n_spins / (time.perf_counter() - t0) / 1e9)
    m, e = sim.observables()
    del sim
    return {"median": statistics.median(rates), "min": min(rates), "max": max(rates),
            "sweeps_per_repeat": n, "m": m, "e": e}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--only", default=None, help="comma list of backends to run")
    args = ap.parse_args()

    import torch

    plan = PLAN_QUICK if args.quick else PLAN_FULL
    min_time = 0.1 if args.quick else 0.5
    repeats = 3 if args.quick else args.repeats
    out = Path(args.out) if args.out else ROOT / "results" / (
        "dev-quick.json" if args.quick else "rtx3070-laptop.json")
    only = set(args.only.split(",")) if args.only else None

    t_start = time.time()
    print("measuring ceilings ...", flush=True)
    ceiling = roofline.measure(quick=args.quick)
    print(json.dumps(ceiling), flush=True)
    bw = ceiling["copy_bandwidth_GBs"]
    ph = ceiling["philox_calls_per_ns"]
    runs = []
    for backend, variant, dim, sizes, algorithm in plan:
        if only and backend not in only:
            continue
        cls = B.load(backend, variant)
        if cls is None:
            print(f"skip {backend}/{variant}: unavailable", flush=True)
            continue
        beta = BETA_C_2D if dim == 2 else BETA_C_3D
        for L in sizes:
            L = max_size(cls, dim) if L == "max" else int(L)
            shape = (L,) * dim
            try:
                r = time_one(cls, shape, algorithm, beta, repeats, min_time)
            except Exception as exc:  # noqa: BLE001 - e.g. out of memory: record and continue
                print(f"FAIL {backend}/{variant} {shape}: {type(exc).__name__}: {exc}"[:300],
                      flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            bpu = cls.bytes_per_update
            eff = r["median"] * bpu
            calls_per_spin = (ceiling["msc_philox_calls_per_word_at_Tc"] / 64.0
                              if cls.stream == "msc" else 0.25)
            if variant == "int8-rng-per-site":
                calls_per_spin = 1.0
            if variant == "msc-eager32":
                calls_per_spin = 16.0 / 64.0
            rng_ceiling = ph / calls_per_spin
            run = {
                "backend": backend, "variant": variant, "algorithm": algorithm,
                "size": list(shape), "dim": dim,
                "dtype": "uint64-packed-1bit" if cls.stream == "msc" else "int8",
                "metric": "flips/ns", "value": round(r["median"], 3), "repeats": repeats,
                "median": round(r["median"], 3), "min": round(r["min"], 3),
                "max": round(r["max"], 3), "sweeps_per_repeat": r["sweeps_per_repeat"],
                "bytes_per_unit": bpu, "effective_GBs": round(eff, 2),
                "pct_copy_bw": round(100 * eff / bw, 2),
                "pct_spec_bw": round(100 * eff / ceiling["spec_bandwidth_GBs"], 2),
                "flops_per_unit": None, "pct_peak_flops": None,
                "memory_ceiling_flips_per_ns": round(bw / bpu, 1),
                "philox_calls_per_update": calls_per_spin,
                "rng_ceiling_flips_per_ns": round(rng_ceiling, 1),
                "pct_roofline": round(100 * r["median"] / min(bw / bpu, rng_ceiling), 2),
                "temperature": 1.0 / beta, "final_m": r["m"], "final_e": r["e"],
                "notes": f"{algorithm}, T=Tc, hot start",
            }
            runs.append(run)
            print(f"{backend:>14}/{variant:<18} {algorithm[:5]} {str(shape):>22}: "
                  f"{run['median']:9.2f} flips/ns [{run['min']:.2f}, {run['max']:.2f}] "
                  f"{run['pct_copy_bw']:6.1f}% copy BW  ({time.time() - t_start:5.0f} s)",
                  flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    res = {"machine": machine_info(), "ceiling": ceiling, "provisional": bool(args.quick),
           "mode": "quick" if args.quick else "full", "runs": runs,
           "seconds": round(time.time() - t_start, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out} in {res['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
