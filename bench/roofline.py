"""Measure the hardware ceilings used by the roofline on the current machine.

* device-to-device copy bandwidth (``torch`` ``copy_``, read + write counted), the
  memory-bound ceiling ("% of STREAM copy");
* Philox4x32-10 throughput (calls/ns) of the same CUDA C function the kernels use, the
  RNG-bound ceiling.

Run directly: ``python bench/roofline.py [--quick]``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SPEC_BANDWIDTH_GBS = 448.0  # RTX 3070 Laptop: 14 Gbps GDDR6 x 256-bit

PHILOX_BENCH = r"""
extern "C" __global__ void philox_bench(unsigned int* out, int calls, unsigned int k0,
                                        unsigned int k1)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int acc = 0;
    for (int i = 0; i < calls; ++i) {
        const u32x4 r = philox4x32_10(tid, (unsigned int)i, 0u, 0u, k0, k1);
        acc ^= r.x ^ r.y ^ r.z ^ r.w;
    }
    out[tid] = acc;
}
"""


def copy_bandwidth(nbytes: int = 1 << 30, repeats: int = 20) -> float:
    """Median device-to-device copy bandwidth in GB/s (bytes read + bytes written)."""
    import torch

    a = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    rates = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        b.copy_(a)
        torch.cuda.synchronize()
        rates.append(2 * nbytes / (time.perf_counter() - t0) / 1e9)
    del a, b
    torch.cuda.empty_cache()
    return statistics.median(rates)


def philox_rate(calls: int = 512, repeats: int = 7) -> float:
    """Median Philox4x32-10 calls per ns over the whole GPU (each call = 4 x 32-bit words)."""
    from pyising.backends.cupy_backend import SOURCE, cp

    mod = cp.RawModule(code=SOURCE.split("#ifdef LH")[0] + PHILOX_BENCH,
                       options=("--std=c++17",))
    fn = mod.get_function("philox_bench")
    sms = cp.cuda.Device().attributes["MultiProcessorCount"]
    threads, blocks = 256, sms * 8 * 4
    out = cp.empty(threads * blocks, dtype=cp.uint32)
    args = (out, cp.int32(calls), cp.uint32(1), cp.uint32(2))
    fn((blocks,), (threads,), args)
    cp.cuda.Device().synchronize()
    rates = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn((blocks,), (threads,), args)
        cp.cuda.Device().synchronize()
        rates.append(threads * blocks * calls / (time.perf_counter() - t0) / 1e9)
    return statistics.median(rates)


def msc_calls_per_word(L: int = 4096, beta: float | None = None) -> float:
    """Mean Philox calls per 64-spin word of the lazy multi-spin kernel at equilibrium."""
    from pyising.backends.cupy_backend import CupyMultiSpin
    from pyising.exact import BETA_C_2D

    sim = CupyMultiSpin((L, L), BETA_C_2D if beta is None else beta, seed=1, init="hot")
    sim.sweep(200)
    return sim.philox_calls_per_word(4)


def measure(quick: bool = False) -> dict:
    """All ceilings as a dict (also stored in the results JSON)."""
    bw = copy_bandwidth(1 << 28 if quick else 1 << 30)
    ph = philox_rate(128 if quick else 512)
    cpw = msc_calls_per_word(2048 if quick else 4096)
    return {
        "copy_bandwidth_GBs": round(bw, 1),
        "spec_bandwidth_GBs": SPEC_BANDWIDTH_GBS,
        "fp32_tflops": None,
        "philox_calls_per_ns": round(ph, 2),
        "msc_philox_calls_per_word_at_Tc": round(cpw, 3),
        "copy_bandwidth_bytes": (1 << 28) if quick else (1 << 30),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    print(json.dumps(measure(ap.parse_args().quick), indent=1))
