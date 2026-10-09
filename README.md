# pyising

**Billions of spin flips per second from Python: 2D/3D Ising Monte Carlo on the GPU across
seven backends, bit-identical results, and a roofline for every rung.**

[![CI](https://github.com/walidhemati/pyising/actions/workflows/ci.yml/badge.svg)](https://github.com/walidhemati/pyising/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/paper.pdf)

<p align="center">
  <img src="docs/media/quench.gif" width="420"
       alt="Domain coarsening after a quench of a 1024x1024 Ising lattice to T = 1.5">
  <br><sub>Domain coarsening after a quench from T = &infin; to T = 1.5 (512&times;512 window of a
  1024&times;1024 lattice, multi-spin-coded CUDA kernel, log-spaced time up to 800 sweeps).</sub>
</p>

## Headline result

<!-- headline:start -->
> **PROVISIONAL**: development run on a shared GPU (`results/dev-quick.json`); final numbers come from `bench/run.py --full`.

| | |
|---|---|
| **Fastest variant** | Triton (msc), L = 8192 |
| **Throughput** | **192.6 spin flips / ns** (2D Metropolis, T = T_c) |
| **% of ceiling** | 20.4 % of measured copy bandwidth (354 GB/s), 16.1 % of nominal (448 GB/s), at 0.375 B/update |
| **Fastest pure-Python-DSL kernel** | Triton (msc): 192.6 flips/ns (20.4 % of copy BW) |
| **GPU** | NVIDIA GeForce RTX 3070 Laptop GPU (single laptop GPU) |
<!-- headline:end -->

## What's inside

| backend | int8 checkerboard (2D/3D, Metropolis + heat bath) | 1-bit multi-spin coding (2D Metropolis) |
|---|---|---|
| NumPy (reference oracle, CI) | yes | yes |
| PyTorch eager | yes | |
| `torch.compile` | yes | |
| Triton | yes | yes |
| NVIDIA Warp | yes | yes |
| Numba-CUDA | yes | yes |
| CUDA C via CuPy `RawModule` (NVRTC) | yes (+ RNG-per-site ablation) | yes (+ eager-32-bit ablation) |

- **Counter-based RNG.** Philox4x32-10 (Random123 known-answer tests pass) keyed by
  (site, sweep, colour, seed), with integer acceptance thresholds: the same seed gives
  **bit-identical spin configurations on every backend**, checked in the test suite.
- **Multi-spin coding.** 64 spins per `uint64`, bit-sliced neighbour counting, and a
  **lazy bitwise comparison** of the uniforms that is exact yet needs only a few Philox calls
  per 64 spins instead of 16.
- **Exact physics checks.** Onsager energy and the *exact finite-torus* energy
  (Kaufman/Ferdinand-Fisher), Onsager-Yang magnetisation, Binder crossings in 2D and 3D (Wolff
  cluster sampler included), 3D T_c vs Ferrenberg, Xu & Landau (2018).
- **Roofline.** Measured copy bandwidth and measured Philox throughput as ceilings; minimal
  traffic model of 3 B/update (int8) and 0.375 B/update (packed bits).

## Quick start

```bash
pip install -e .[gpu]          # or .[triton], .[warp], .[numba], .[cupy]; plain `pip install -e .` = NumPy only
pyising list                   # which backends/variants run on this machine
pyising run --backend triton --variant msc --L 8192 --T 2.0 --sweeps 1000
pyising validate --quick       # physics + bitwise-identity checks (~30 s)
python bench/run.py --quick    # smoke benchmark (< 2 min)
```

```python
from pyising import create, TC_2D
sim = create("cupy-cuda-c", "msc", shape=(16384, 16384), beta=1 / TC_2D, seed=1)
sim.sweep(100)
m, e = sim.observables()
```

## Validation

![validation](figures/validation.png)

![binder](figures/binder.png)

Checked against: the exact energy of the finite L&times;L torus at every temperature, the
Onsager-Yang spontaneous magnetisation, Onsager's T_c (2D Binder crossing), the simple-cubic
K_c = 0.221654626(5) of Ferrenberg, Xu & Landau (3D Binder crossing), and Wolff cluster
simulations (3D energies). Summary from `results/validation.json`:

<!-- validation:start -->
> quick validation run (small lattices)

| check | result |
|---|---|
| bitwise identity across backends | 33 checks, all identical |
| 2D energy vs exact finite torus | max abs error 2.1e-02, max 7.87 sigma |
| 2D abs(m) vs Onsager-Yang (T < T_c - 0.2) | max abs error 2.9e-04 |
| 2D T_c from Binder crossing | 2.2288 (1.78 % from Onsager) |
| 3D T_c from Binder crossing | 4.4705 (0.91 % from Ferrenberg et al. 2018) |
| 3D GPU vs Wolff energies | max 1.06 sigma |
<!-- validation:end -->

## Performance

![ladder](figures/ladder.png)

![scaling](figures/scaling.png)

<!-- table:start -->
Source: `results/dev-quick.json`.

| backend | variant | L | flips/ns | eff. GB/s | % of copy BW |
|---|---|---:|---:|---:|---:|
| NumPy | int8 | 256 | 0.06 | 0.2 | 0.1 |
| torch eager | int8 | 1024 | 0.16 | 0.5 | 0.1 |
| torch.compile | int8 | 1024 | 1.08 | 3.2 | 0.9 |
| Triton | int8 | 4096 | 35.42 | 106.2 | 30.0 |
| CUDA C (CuPy) | int8 | 1024 | 46.48 | 139.4 | 39.4 |
| Triton | msc | 8192 | 192.57 | 72.2 | 20.4 |
| CUDA C (CuPy) | msc | 8192 | 150.30 | 56.4 | 15.9 |
| CUDA C (CuPy) | msc-eager32 | 8192 | 76.74 | 28.8 | 8.1 |
<!-- table:end -->

## How it works

1. **NumPy reference**: compressed checkerboard layout (one array per colour), vectorised
   Philox, integer thresholds. This is the oracle every other backend must match bit for bit.
2. **PyTorch eager / `torch.compile`**: the same algorithm as tensor ops; compile fuses it.
3. **Triton, Warp, Numba-CUDA**: hand-written kernels in Python; one Philox call serves 4 sites.
4. **CUDA C (CuPy/NVRTC)**: the hand-tuned reference (`char4` loads, compile-time sizes).
5. **Multi-spin coding** (Triton, Warp, Numba, CUDA C): 64 spins per word, 8x less traffic;
   the bottleneck moves to the RNG, which the lazy bitwise comparison attacks.

Memory model: a half sweep reads both colour arrays once and writes one, so an update costs
3 spin-sizes of DRAM traffic. Ceilings: `bench/roofline.py` (copy bandwidth, Philox calls/ns).

## Reproduce the paper

```bash
python bench/run.py --full          # < 15 min on an idle GPU -> results/rtx3070-laptop.json
python -m pyising validate          # -> results/validation.json
python figures/make_figures.py      # figures/*.pdf|png, paper/results.tex, README tables
cd paper && tectonic paper.tex      # paper/paper.pdf
```

## Citation

```bibtex
@software{hemati_pyising_2026,
  author  = {Hemati, Walid},
  title   = {pyising: roofline-optimised Ising Monte Carlo on the GPU from Python},
  year    = {2026},
  url     = {https://github.com/walidhemati/pyising},
  license = {MIT}
}
```

MIT licence, see [LICENSE](LICENSE). Metadata in [CITATION.cff](CITATION.cff).

---
<sub>Part of the **Python-HPC suite**:
[pylbm](https://github.com/walidhemati/pylbm) ·
[pyheat3d](https://github.com/walidhemati/pyheat3d) ·
[pygravity](https://github.com/walidhemati/pygravity) ·
[pymd](https://github.com/walidhemati/pymd) ·
[pyising](https://github.com/walidhemati/pyising) ·
[pyspectral](https://github.com/walidhemati/pyspectral)</sub>
