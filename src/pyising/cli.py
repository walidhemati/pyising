"""Command line interface: ``pyising run | validate | bench | list | movie``."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import backends as B
from .exact import TC_2D, TC_3D


def _sim(args) -> B.IsingSimulation:
    shape = (args.L,) * args.dim
    T = args.T if args.T is not None else (TC_2D if args.dim == 2 else TC_3D)
    return B.create(args.backend, args.variant, shape=shape, beta=1.0 / T, seed=args.seed,
                    algorithm=args.algorithm, init=args.init)


def cmd_list(args) -> int:
    for be, var in B.REGISTRY:
        cls = B.load(be, var)
        status = "ok" if cls is not None else "unavailable"
        extra = f"dims={cls.dims} algs={cls.algorithms} B/update={cls.bytes_per_update:g}" \
            if cls else ""
        print(f"{be:>14} / {var:<18} {status:<12} {extra}")
    return 0


def cmd_run(args) -> int:
    sim = _sim(args)
    sim.sweep(args.warmup)
    sim.synchronize()
    t0 = time.perf_counter()
    sim.sweep(args.sweeps)
    sim.synchronize()
    dt = time.perf_counter() - t0
    m, e = sim.observables()
    rate = args.sweeps * sim.n_spins / dt / 1e9
    print(f"{sim.backend}/{sim.variant} {sim.algorithm} shape={sim.shape} T={1 / sim.beta:.5f}")
    print(f"m = {m:+.6f}   e = {e:.6f}   {rate:.2f} flips/ns ({dt:.3f} s, single run)")
    return 0


def cmd_bench(args) -> int:
    """Quick single-configuration timing (publishable numbers come from bench/run.py)."""
    sim = _sim(args)
    sim.sweep(args.warmup)
    sim.synchronize()
    rates = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        sim.sweep(args.sweeps)
        sim.synchronize()
        rates.append(args.sweeps * sim.n_spins / (time.perf_counter() - t0) / 1e9)
    print(f"{sim.backend}/{sim.variant} L={args.L} dim={args.dim}: median "
          f"{np.median(rates):.2f} flips/ns (min {min(rates):.2f}, max {max(rates):.2f}); "
          f"use bench/run.py for publishable numbers")
    return 0


def cmd_validate(args) -> int:
    from .validate import run

    res = run(quick=args.quick, out=args.out)
    s = res["summary"]
    for k, v in s.items():
        print(f"{k:>40}: {v}")
    return 0 if s["all_identical"] else 1


def cmd_movie(args) -> int:
    from .movie import quench_gif

    quench_gif(args.out, L=args.L, T=args.T or 1.5, frames=args.frames, backend=args.backend,
               variant=args.variant)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pyising", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, sweeps: int = 100) -> None:
        sp.add_argument("--backend", default="cupy-cuda-c", choices=B.BACKENDS)
        sp.add_argument("--variant", default="int8")
        sp.add_argument("--dim", type=int, default=2, choices=(2, 3))
        sp.add_argument("--L", type=int, default=1024)
        sp.add_argument("--T", type=float, default=None, help="temperature (default T_c)")
        sp.add_argument("--algorithm", default="metropolis", choices=("metropolis", "heatbath"))
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--init", default="hot", choices=("hot", "cold"))
        sp.add_argument("--sweeps", type=int, default=sweeps)
        sp.add_argument("--warmup", type=int, default=10)

    sub.add_parser("list", help="list backends/variants and availability")
    common(sub.add_parser("run", help="run a simulation and print m, e, throughput"))
    sb = sub.add_parser("bench", help="quick timing of one configuration")
    common(sb)
    sb.add_argument("--repeats", type=int, default=5)
    sv = sub.add_parser("validate", help="physics validation suite")
    sv.add_argument("--quick", action="store_true")
    sv.add_argument("--out", default="results/validation.json")
    sm = sub.add_parser("movie", help="GIF of domain coarsening after a quench")
    sm.add_argument("--out", default="docs/media/quench.gif")
    sm.add_argument("--L", type=int, default=512)
    sm.add_argument("--T", type=float, default=1.5)
    sm.add_argument("--frames", type=int, default=60)
    sm.add_argument("--backend", default="cupy-cuda-c")
    sm.add_argument("--variant", default="msc")
    args = p.parse_args(argv)
    return {"list": cmd_list, "run": cmd_run, "bench": cmd_bench, "validate": cmd_validate,
            "movie": cmd_movie}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
