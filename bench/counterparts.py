"""Same-GPU head-to-head with NVIDIA's reference Ising codes (Romero et al., CPC 2020).

    python bench/counterparts.py --quick   # smoke run, L = 4096, < 60 s, < 1 GB  (provisional)
    python bench/counterparts.py --full    # publishable run, idle GPU, L up to 131072

Both modes write ``results/counterparts.json``; ``--quick`` marks it ``"provisional": true``.

Counterpart: https://github.com/NVIDIA/ising-gpu, pinned to ``PINNED_COMMIT`` and checked out in
``~/python-hpc/_ref/ising-gpu`` (``--ref-dir``; cloned if missing). Five of its programs simulate
the 2D Ising model with checkerboard Metropolis updates:

* ``basic_cuda``    int8 spins, cuRAND Philox uniforms written to a float array, ``exp`` per site;
* ``basic_python``  the same algorithm with Numba kernels and CuPy cuRAND (our Python peer);
* ``tensorcore``    fp16 spins, neighbour sums as cuBLAS tensor-core GEMMs;
* ``optimized_old`` multi-spin coding with 4 bits per spin (the version in the CPC paper);
* ``cuIsingModel``  current optimised code: 1 bit per spin, random bits from a 16-entry table.

The source is never modified. Build-time patches (all needed for CUDA 13 / sm_86, documented in
``BUILDS`` and in the JSON): ``-arch=sm_86`` instead of sm_70/sm_89, ``-std=c++17`` for CCCL 3,
and a force-included header (``bench/counterparts/*.h``) that restores two runtime-API overloads
removed in CUDA 13 (and, for ``tensorcore``, a half->double CUB sum used only after timing).
``basic_python`` runs in its own venv (``~/.venvs/cp-ising``) through a wrapper that only aliases
the module ``cupy.cuda.curand`` to its current location ``cupy_backends.cuda.libs.curand``.

Metric (identical for every code): attempted spin updates per ns = L^2 x sweeps / seconds, one
sweep = both colours. We parse the number each counterpart prints itself (``updates per ns``,
``flips/ns`` or ``spins/ns``); their timed regions differ slightly (see ``TIMING`` notes) and we
do not change them. All runs at T = T_c (``-a 1.0``; their T_c = 2.26918531421), hot start,
J = 1, periodic boundaries. Each (code, L) is run once to size the sweep count so that a timed
run lasts >= ``min_time``, then ``repeats`` times; we report median/min/max.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as bench_run  # noqa: E402

from pyising import backends as B  # noqa: E402
from pyising.exact import BETA_C_2D  # noqa: E402

REPO_URL = "https://github.com/NVIDIA/ising-gpu.git"
PINNED_COMMIT = "935796d7a26016670363af7a0dced8a9ebcd4714"
DEFAULT_REF = ROOT.parent / "_ref" / "ising-gpu"
DEFAULT_CP_PYTHON = Path.home() / ".venvs" / "cp-ising" / "bin" / "python"
CUDA_HOME = Path("/usr/local/cuda")
SHIMS = Path(__file__).resolve().parent / "counterparts"
OUT = ROOT / "results" / "counterparts.json"

QUICK_SIZES = [4096]
FULL_SIZES = [4096, 8192, 16384, 32768, 65536, 131072]

# Runs `basic_python/ising_basic.py` unchanged after aliasing the module CuPy moved in v9+.
PY_WRAPPER = (
    "import runpy, sys\n"
    "import cupy_backends.cuda.libs.curand as _curand\n"
    "sys.modules['cupy.cuda.curand'] = _curand\n"
    "script = sys.argv[1]\n"
    "sys.argv = sys.argv[1:]\n"
    "runpy.run_path(script, run_name='__main__')\n"
)


@dataclass
class Counterpart:
    """One program of NVIDIA/ising-gpu: how to build it, run it and read its throughput."""

    name: str
    subdir: str
    binary: str
    spin_bits: float
    bytes_per_spin_device: float  # device memory per spin (for the memory guard)
    max_L: int
    rate_re: str
    obs_re: str | None
    obs_name: str | None
    timing: str
    make_vars: dict[str, str] = field(default_factory=dict)
    patches: list[str] = field(default_factory=list)

    def args(self, L: int, sweeps: int, seed: int) -> list[str]:
        warm = str(min(100, max(10, sweeps // 5)))
        if self.name in ("basic_cuda", "basic_python"):
            return ["-x", str(L), "-y", str(L), "-w", warm, "-n", str(sweeps), "-a", "1.0",
                    "-s", str(seed)]
        if self.name == "tensorcore":
            return ["-x", str(L // 256), "-y", str(L // 256), "-w", warm, "-n", str(sweeps),
                    "-a", "1.0", "-s", str(seed)]
        return ["-x", str(L), "-y", str(L), "-n", str(sweeps), "-a", "1.0", "-s", str(seed)]


def _flags(*parts: str) -> str:
    return " ".join(parts)


ARCH = "-arch=sm_86"
BUILDS: list[Counterpart] = [
    Counterpart(
        "basic_cuda", "basic_cuda", "ising_basic", 8, 3.0, 32768,
        r"updates per ns:\s*([0-9.eE+-]+)", r"average magnetism \(absolute\):\s*([0-9.eE+-]+)",
        "abs_m",
        "host std::chrono around the trial loop (warm-up excluded); includes the cuRAND call "
        "that fills one float per site before each half-sweep",
        {"CUDACFLAGS": _flags("-std=c++17 -c -O3 -lineinfo", ARCH, "-Xptxas=-v",
                              "-I../external/cub")},
        ["-arch=sm_70 -> sm_86 (sm_70 is not supported by CUDA 13)",
         "-std=c++11 -> c++17 (CCCL 3 / CUB in CUDA 13 requires C++17)"],
    ),
    Counterpart(
        "basic_python", "basic_python", "ising_basic.py", 8, 3.0, 32768,
        r"updates per ns:\s*([0-9.eE+-]+)", r"average magnetism \(absolute\):\s*([0-9.eE+-]+)",
        "abs_m",
        "host time.time() around the trial loop (warm-up excluded); includes the CuPy cuRAND "
        "call that fills one float per site before each half-sweep",
        {},
        ["run through a wrapper that aliases sys.modules['cupy.cuda.curand'] to "
         "cupy_backends.cuda.libs.curand (module moved in CuPy >= 9); script unchanged",
         "separate venv ~/.venvs/cp-ising: numba-cuda (cuda-bindings 12.9.4, whose IPC handle "
         "still pickles), cupy-cuda12x 12.3.0, mpi4py + mpich, numpy 1.26"],
    ),
    Counterpart(
        "tensorcore", "tensorcore", "ising_tensorcore", 16, 4.0, 16384,
        r"updates per ns:\s*([0-9.eE+-]+)", r"average magnetism \(absolute\):\s*([0-9.eE+-]+)",
        "abs_m",
        "host std::chrono around the trial loop (warm-up excluded)",
        {"CUDACFLAGS": _flags("-std=c++17 -c -O3 -lineinfo", ARCH, "-Xptxas=-v",
                              "-I../external/cub",
                              f"--pre-include {SHIMS / 'tensorcore_compat.h'}")},
        ["-arch=sm_70 -> sm_86", "-std=c++11 -> c++17",
         "--pre-include tensorcore_compat.h: cudaMemAdvise(int device) overload removed in "
         "CUDA 13; cub::DeviceReduce::Sum(__half*, double*) used only for the final |m| "
         "printout after timing"],
    ),
    Counterpart(
        "optimized_old", "optimized/old", "cuIsing", 4, 0.5, 65536,
        r"([0-9.]+) flips/ns", r"Final\s+magnetization:\s*([0-9.eE+-]+)", "abs_m",
        "cudaEvent pair around the update loop (single GPU)",
        {"CUDACFLAGS": _flags("-c -O3 --use_fast_math -lineinfo", ARCH, "-Xptxas=-v",
                              f"--pre-include {SHIMS / 'cuda13_compat.h'}")},
        ["-arch=sm_70 -> sm_86",
         "--pre-include cuda13_compat.h: cudaMemAdvise(int device) overload removed in CUDA 13"],
    ),
    Counterpart(
        "cuIsingModel", "optimized/cuIsingModel", "cuIsing", 1, 0.125, 131072,
        r"spins/ns:\s*([0-9.eE+-]+)", r"Final energy:\s*([0-9.eE+-]+)", "e",
        "host wall clock around the update loop, which synchronises the device after every "
        "half-sweep; value = the final 'spins/ns' it prints",
        {"CUDA_HOME": str(CUDA_HOME),
         "CUDACFLAGS": _flags("-c -O3 -lineinfo -gencode arch=compute_86,code=sm_86",
                              "-Xptxas=-v", f"--pre-include {SHIMS / 'cuda13_compat.h'}")},
        ["SMS=89 -> sm_86, CUDA_HOME=/opt/cuda-12.8.1 -> /usr/local/cuda",
         "--pre-include cuda13_compat.h: cudaMemAdvise(int) and cudaMemPrefetchAsync(int) "
         "overloads removed in CUDA 13"],
    ),
]
CP = {c.name: c for c in BUILDS}

#: pyising variants matched to the counterparts (int8 <-> basic/tensorcore, msc <-> optimised).
PY_QUICK = [("triton", "int8"), ("cupy-cuda-c", "int8"), ("warp", "msc"), ("cupy-cuda-c", "msc")]
PY_FULL = [(b, v) for v in ("int8", "msc") for b in ("triton", "warp", "numba", "cupy-cuda-c")]
PY_MAX_L = {"int8": 32768, "msc": 131072}


# ------------------------------------------------------------------------------- helpers --
def sh(cmd: list[str], cwd: Path | None = None, env: dict | None = None,
       timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout, capture_output=True,
                          text=True, check=False)


def tool_version(cmd: list[str]) -> str | None:
    try:
        out = sh(cmd, timeout=30).stdout.strip().splitlines()
        return out[-1] if cmd[0].endswith("nvcc") else out[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        return None


def ensure_checkout(ref: Path) -> str:
    """Clone the counterpart if needed; return the checked-out commit (warn if not pinned)."""
    if not (ref / ".git").exists():
        ref.parent.mkdir(parents=True, exist_ok=True)
        r = sh(["git", "clone", REPO_URL, str(ref)])
        if r.returncode:
            raise SystemExit(f"git clone failed: {r.stderr}")
        sh(["git", "checkout", "--quiet", PINNED_COMMIT], cwd=ref)
    head = sh(["git", "rev-parse", "HEAD"], cwd=ref).stdout.strip()
    if head != PINNED_COMMIT:
        print(f"WARNING: {ref} is at {head}, pinned commit is {PINNED_COMMIT}", flush=True)
    dirty = sh(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ref).stdout.strip()
    if dirty:
        raise SystemExit(f"counterpart sources in {ref} are modified; refusing to benchmark:\n"
                         f"{dirty}")
    return head


def build_env() -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{CUDA_HOME / 'bin'}:{env.get('PATH', '')}"
    return env


def build(ref: Path, cp: Counterpart, rebuild: bool) -> str | None:
    """Build one counterpart with make (build-flag overrides only). Return an error or None."""
    if cp.name == "basic_python":
        return None
    d = ref / cp.subdir
    if (d / cp.binary).exists() and not rebuild:
        return None
    sh(["make", "clean"], cwd=d, env=build_env())
    cmd = ["make", "-j4"] + [f"{k}={v}" for k, v in cp.make_vars.items()]
    r = sh(cmd, cwd=d, env=build_env(), timeout=1800)
    if r.returncode or not (d / cp.binary).exists():
        return (r.stdout + r.stderr)[-2000:]
    return None


def cp_python_env(cp_python: Path) -> dict:
    """Environment for the counterpart venv: pip CUDA 12 libraries on the loader path."""
    env = dict(os.environ)
    site = sorted(cp_python.parent.parent.glob("lib/python3*/site-packages"))
    libs = [str(p) for s in site for p in sorted((s / "nvidia").glob("*/lib"))]
    env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    return env


def command(ref: Path, cp: Counterpart, L: int, sweeps: int, seed: int,
            cp_python: Path) -> tuple[list[str], dict]:
    args = cp.args(L, sweeps, seed)
    if cp.name == "basic_python":
        return ([str(cp_python), "-c", PY_WRAPPER, str(ref / cp.subdir / cp.binary)] + args,
                cp_python_env(cp_python))
    env = dict(os.environ)
    # Load every kernel at start-up: with CUDA's default lazy loading the first launch inside
    # a counterpart's timed loop would include module loading (optimized_old has no warm-up).
    env["CUDA_MODULE_LOADING"] = "EAGER"
    return [str(ref / cp.subdir / cp.binary)] + args, env


def run_cp(ref: Path, cp: Counterpart, L: int, sweeps: int, seed: int, cp_python: Path,
           timeout: float) -> tuple[float, float | None, str]:
    cmd, env = command(ref, cp, L, sweeps, seed, cp_python)
    r = sh(cmd, cwd=Path(tempfile.gettempdir()), env=env, timeout=timeout)
    out = r.stdout + r.stderr
    m = re.findall(cp.rate_re, out)
    if r.returncode or not m:
        raise RuntimeError(f"exit {r.returncode}: {out[-600:]}")
    obs = re.findall(cp.obs_re, out) if cp.obs_re else []
    shown = shlex.join([f"{cp.subdir}/{cp.binary}"] + cp.args(L, sweeps, seed))
    return float(m[-1]), (float(obs[-1]) if obs else None), shown


def free_bytes() -> int:
    import torch

    return torch.cuda.mem_get_info()[0]


# ---------------------------------------------------------------------------------- runs --
def time_counterpart(ref: Path, cp: Counterpart, L: int, args, cp_python: Path) -> dict:
    seed = 12345
    n0 = max(4, math.ceil(2e8 / L**2))
    rate0, _, _ = run_cp(ref, cp, L, n0, seed, cp_python, args.timeout)  # sizing + warm-up
    sweeps = max(10, math.ceil(args.min_time * rate0 * 1e9 / L**2))
    rates, obs = [], []
    cmd = ""
    for _ in range(args.repeats):
        rate, o, cmd = run_cp(ref, cp, L, sweeps, seed, cp_python, args.timeout)
        rates.append(rate)
        obs.append(o)
    return {
        "code": "NVIDIA/ising-gpu", "variant": cp.name, "L": L, "size": [L, L],
        "spin_bits": cp.spin_bits, "metric": "flips/ns",
        "median": round(statistics.median(rates), 3), "min": round(min(rates), 3),
        "max": round(max(rates), 3), "repeats": len(rates), "raw": [round(x, 3) for x in rates],
        "sweeps_per_repeat": sweeps, "temperature": 1.0 / BETA_C_2D,
        "observable": cp.obs_name, "observable_values": obs, "command": cmd,
        "notes": f"T=Tc, hot start, their timing: {cp.timing}",
    }


def time_pyising(backend: str, variant: str, L: int, args) -> dict:
    cls = B.load(backend, variant)
    if cls is None:
        raise RuntimeError("backend unavailable")
    r = bench_run.time_one(cls, (L, L), "metropolis", BETA_C_2D, args.repeats, args.min_time)
    return {
        "code": "pyising", "variant": f"{backend}/{variant}", "backend": backend,
        "layout": variant, "L": L, "size": [L, L], "spin_bits": 1 if variant == "msc" else 8,
        "metric": "flips/ns", "median": round(r["median"], 3), "min": round(r["min"], 3),
        "max": round(r["max"], 3), "repeats": args.repeats,
        "sweeps_per_repeat": r["sweeps_per_repeat"], "temperature": 1.0 / BETA_C_2D,
        "observable": "e", "observable_values": [r["e"]],
        "notes": "T=Tc, hot start, bench/run.py time_one: JIT warm-up, device sync around "
                 "each timed repeat",
    }


def physics_check(ref: Path, L: int, sweeps: int, seeds: list[int], args) -> dict:
    """Energy per spin after ``sweeps`` sweeps from a hot start at T_c, several seeds.

    Same model + same temperature + same sweep semantics => the same relaxation curve e(t)
    within statistical scatter, whatever the random stream.
    """
    out: dict = {"L": L, "sweeps": sweeps, "temperature": 1.0 / BETA_C_2D, "seeds": seeds,
                 "energy_per_spin": {}}
    cp = CP["cuIsingModel"]
    vals = []
    for s in seeds:
        _, e, _ = run_cp(ref, cp, L, sweeps, s, args.cp_python, args.timeout)
        vals.append(e)
    out["energy_per_spin"]["NVIDIA/ising-gpu cuIsingModel"] = vals
    for backend, variant in (("cupy-cuda-c", "msc"), ("cupy-cuda-c", "int8")):
        cls = B.load(backend, variant)
        if cls is None:
            continue
        vals = []
        for s in seeds:
            sim = cls((L, L), BETA_C_2D, seed=s, algorithm="metropolis", init="hot")
            sim.sweep(sweeps)
            sim.synchronize()
            vals.append(float(sim.observables()[1]))
            del sim
        out["energy_per_spin"][f"pyising {backend}/{variant}"] = vals
    stats = {k: (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
             for k, v in out["energy_per_spin"].items()}
    out["mean_std"] = {k: [round(m, 6), round(s, 6)] for k, (m, s) in stats.items()}
    ref_key = "NVIDIA/ising-gpu cuIsingModel"
    m0, s0 = stats[ref_key]
    out["z_vs_cuIsingModel"] = {
        k: round((m - m0) / math.sqrt((s**2 + s0**2) / len(seeds) + 1e-30), 2)
        for k, (m, s) in stats.items() if k != ref_key}
    return out


def identity_check(ref: Path, args, L: int = 2048, sweeps: int = 100, seed: int = 1234) -> dict:
    """basic_cuda and basic_python share cuRAND Philox, seed, layout and update rule, so the
    same command line must give the same final |m| (C++ prints 6 significant digits)."""
    m = {name: run_cp(ref, CP[name], L, sweeps, seed, args.cp_python, args.timeout)[1]
         for name in ("basic_cuda", "basic_python")}
    a, b = m["basic_cuda"], m["basic_python"]
    return {"L": L, "sweeps": sweeps, "seed": seed, "abs_m": m,
            "identical_to_printed_digits": abs(a - b) <= 1e-5 * max(abs(a), abs(b), 1e-12)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true", help="L=4096 only, short runs (provisional)")
    g.add_argument("--full", action="store_true", help="publishable run (idle GPU only)")
    ap.add_argument("--ref-dir", type=Path, default=DEFAULT_REF)
    ap.add_argument("--cp-python", type=Path, default=DEFAULT_CP_PYTHON,
                    help="python of the venv for basic_python (numba-cuda, cupy, mpi4py)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rebuild", action="store_true", help="make clean + rebuild counterparts")
    ap.add_argument("--only", default=None, help="comma list of counterpart/pyising names")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.min_time = 0.1 if args.quick else 0.5
    args.timeout = 60.0 if args.quick else 600.0
    sizes = QUICK_SIZES if args.quick else FULL_SIZES
    py_plan = PY_QUICK if args.quick else PY_FULL
    mem_cap = 1e9 if args.quick else float("inf")
    only = set(args.only.split(",")) if args.only else None

    import torch

    t_start = time.time()
    commit = ensure_checkout(args.ref_dir)
    build_info, skipped = {}, {}
    for cp in BUILDS:
        if only and cp.name not in only:
            continue
        err = build(args.ref_dir, cp, args.rebuild)
        if cp.name == "basic_python" and not args.cp_python.exists():
            err = f"{args.cp_python} not found (create the cp-ising venv, see docstring)"
        if err:
            skipped[cp.name] = err
            print(f"skip {cp.name}: {err[-300:]}", flush=True)
        build_info[cp.name] = {"dir": cp.subdir, "make_overrides": cp.make_vars,
                               "patches": cp.patches, "timing": cp.timing,
                               "spin_bits": cp.spin_bits}
    print(f"builds ready ({time.time() - t_start:.0f} s)", flush=True)

    runs, failures = [], []
    for L in sizes:
        for cp in BUILDS:
            if (only and cp.name not in only) or cp.name in skipped or L > cp.max_L:
                continue
            need = L * L * cp.bytes_per_spin_device * 1.1 + 64e6
            if need > min(mem_cap, 0.8 * free_bytes()):
                print(f"skip {cp.name} L={L}: needs {need / 1e9:.2f} GB", flush=True)
                continue
            try:
                r = time_counterpart(args.ref_dir, cp, L, args, args.cp_python)
            except (RuntimeError, subprocess.SubprocessError) as exc:
                failures.append({"variant": cp.name, "L": L, "error": str(exc)[-400:]})
                print(f"FAIL {cp.name} L={L}: {str(exc)[-200:]}", flush=True)
                continue
            runs.append(r)
            print(f"{'ising-gpu/' + cp.name:>26} L={L:>6}: {r['median']:9.2f} flips/ns "
                  f"[{r['min']:.2f}, {r['max']:.2f}] x{r['sweeps_per_repeat']} sweeps "
                  f"({time.time() - t_start:4.0f} s)", flush=True)
        for backend, variant in py_plan:
            name = f"{backend}/{variant}"
            if (only and name not in only and backend not in only) or L > PY_MAX_L[variant]:
                continue
            bpu = 1.0 if variant == "int8" else 0.125
            if L * L * bpu * 1.1 + 64e6 > min(mem_cap, 0.8 * free_bytes()):
                continue
            try:
                r = time_pyising(backend, variant, L, args)
            except Exception as exc:  # noqa: BLE001 - record (e.g. out of memory) and go on
                failures.append({"variant": name, "L": L, "error": str(exc)[-400:]})
                print(f"FAIL pyising {name} L={L}: {exc}"[:300], flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            runs.append(r)
            print(f"{'pyising/' + name:>26} L={L:>6}: {r['median']:9.2f} flips/ns "
                  f"[{r['min']:.2f}, {r['max']:.2f}] x{r['sweeps_per_repeat']} sweeps "
                  f"({time.time() - t_start:4.0f} s)", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    checks: dict = {}
    if not {"basic_cuda", "basic_python"} & set(skipped) and (
            not only or {"basic_cuda", "basic_python"} <= only):
        try:
            checks["basic_cuda_vs_basic_python"] = identity_check(args.ref_dir, args)
        except (RuntimeError, subprocess.SubprocessError) as exc:
            failures.append({"variant": "identity_check", "error": str(exc)[-400:]})
    if "cuIsingModel" not in skipped and (not only or "cuIsingModel" in only):
        try:
            checks["energy_relaxation"] = physics_check(
                args.ref_dir, 4096, 200 if args.quick else 1000,
                # 12 seeds in --full: three seeds under-resolve the scatter of e(t)
                [1, 2, 3] if args.quick else list(range(101, 113)), args)
        except (RuntimeError, subprocess.SubprocessError) as exc:
            failures.append({"variant": "physics_check", "error": str(exc)[-400:]})

    machine = bench_run.machine_info()
    machine["nvcc"] = tool_version([str(CUDA_HOME / "bin" / "nvcc"), "--version"])
    machine["gcc"] = tool_version(["gcc", "--version"])
    if args.cp_python.exists():
        r = sh([str(args.cp_python), "-c",
                "import importlib.metadata as m; print({p: m.version(p) for p in "
                "('numba', 'numba-cuda', 'cuda-bindings', 'cupy-cuda12x', 'numpy', 'mpi4py')})"],
               env=cp_python_env(args.cp_python), timeout=60)
        machine["cp_ising_venv"] = r.stdout.strip() or None
    mode = "quick" if args.quick else "full"
    res = {
        "machine": machine, "provisional": mode != "full", "mode": mode,
        "counterpart": {"repo": REPO_URL, "commit": commit, "pinned_commit": PINNED_COMMIT,
                        "paper": "Romero, Bisson, Fatica, Bernaschi, Comput. Phys. Commun. 256 "
                                 "(2020) 107473, doi:10.1016/j.cpc.2020.107473",
                        "env": {"CUDA_MODULE_LOADING": "EAGER (CUDA binaries)"},
                        "builds": build_info, "skipped": skipped},
        "metric": "attempted spin updates per ns over the whole lattice (L^2 x sweeps / s), "
                  "as printed by each counterpart; pyising: bench/run.py time_one",
        "temperature": 1.0 / BETA_C_2D, "min_time_s": args.min_time,
        "runs": runs, "failures": failures, "checks": checks,
        "seconds": round(time.time() - t_start, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out} in {res['seconds']} s")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # same reason as bench/run.py: skip a teardown that can hang
