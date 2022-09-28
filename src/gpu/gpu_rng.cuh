// TODO: implement cuRAND
#ifndef __GPU_RNG_CUH__
#define __GPU_RNG_CUH__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include "common/switcher.h"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

namespace detail_ {
template <typename F>
__global__ void SingleKernel(F func) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid <= 1) func();
}
};  // namespace detail_
template <typename F>
inline void Execute(F &&func) {
  auto kernel = &detail_::SingleKernel<F>;
  kernel<<<1, 1>>>(func);
}

template <typename T>
inline T *AllocateGPUInstance(uint64_t seed = 0) {
  T *result;
  cudaMalloc(&result, sizeof(T));
  auto constructor = [=](T *ptr) __device__ { new (ptr) T(seed); };
  auto kernel      = &detail_::SingleKernel<decltype(constructor)>;
  kernel<<<1, 1>>>(constructor);
}

class RandomGPU {
public:
  F_GPU_ONLY explicit RandomGPU(uint64_t seed = 0) { init_cuRAND(seed); }

private:
  F_GPU_ONLY
  void init_cuRAND(uint64_t seed) { curand_init(seed, 0, 0, &state); }

  curandState state;
};

FLG_NAMESPACE_END

#endif
