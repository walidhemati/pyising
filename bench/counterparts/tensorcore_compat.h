/* Extra build-time shim for NVIDIA/ising-gpu/tensorcore under CUDA >= 13 (CCCL 3).
 *
 * After the timed loop, tensorcore/main.cu reduces the __half lattice into a double with
 * cub::DeviceReduce::Sum(__half*, double*, long long). CCCL 3 no longer provides that
 * mixed-type instantiation. This explicit specialisation (declared before main.cu is parsed,
 * via `nvcc --pre-include`) computes the same sum with a plain grid-stride kernel and double
 * atomics. It is only used for the final |m| printout, never inside the timed region, and it
 * leaves the counterpart's source and its update algorithm unchanged.
 */
#pragma once
#include "cuda13_compat.h"
#include <cuda_fp16.h>
#include <cub/cub.cuh>

__global__ void cp_compat_half_sum_k(const __half *in, double *out, long long n) {
  double acc = 0.0;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
       i += (long long)gridDim.x * blockDim.x) {
    acc += (double)__half2float(in[i]);
  }
  atomicAdd(out, acc);
}

template <>
inline cudaError_t cub::DeviceReduce::Sum<__half *, double *, long long>(
    void *d_temp_storage, size_t &temp_storage_bytes, __half *d_in, double *d_out,
    long long num_items, cudaStream_t stream) {
  if (d_temp_storage == nullptr) {
    temp_storage_bytes = 1;
    return cudaSuccess;
  }
  cudaError_t err = cudaMemsetAsync(d_out, 0, sizeof(double), stream);
  if (err != cudaSuccess) return err;
  cp_compat_half_sum_k<<<1024, 256, 0, stream>>>(d_in, d_out, num_items);
  return cudaGetLastError();
}
