/* Build-time compatibility shim for NVIDIA/ising-gpu under CUDA >= 13.
 *
 * CUDA 13 removed the cudaMemAdvise(ptr, bytes, advice, int device) overload; only the
 * cudaMemLocation form remains. The counterpart sources are left untouched: this header is
 * force-included with `nvcc --pre-include` and restores the old overload by forwarding the
 * device id as a cudaMemLocation. It does not touch any kernel or the simulated algorithm.
 */
#pragma once
#include <cuda_runtime.h>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
static inline cudaError_t cudaMemAdvise(const void *ptr, size_t count, cudaMemoryAdvise advice,
                                        int device) {
  cudaMemLocation loc;
  loc.type = (device == cudaCpuDeviceId) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
  loc.id = (device == cudaCpuDeviceId) ? 0 : device;
  return cudaMemAdvise(ptr, count, advice, loc);
}

/* Same for cudaMemPrefetchAsync(ptr, bytes, int device, stream). */
static inline cudaError_t cudaMemPrefetchAsync(const void *ptr, size_t count, int device,
                                               cudaStream_t stream = 0) {
  cudaMemLocation loc;
  loc.type = (device == cudaCpuDeviceId) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
  loc.id = (device == cudaCpuDeviceId) ? 0 : device;
  return cudaMemPrefetchAsync(ptr, count, loc, 0, stream);
}
#endif
