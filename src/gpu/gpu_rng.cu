#include "gpu_rng.hpp"

#include "fledge.h"

FLG_NAMESPACE_BEGIN

namespace detail_ {
template <typename... Args>
__global__ void curand_init_Kernel(Args... args) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= 1) return;
  curand_init(std::forward<Args>(args)...);
}
}  // namespace detail_

F_GPU_ONLY
Float RandomGPU::get1D() {
  return curand_uniform(&state);
}

F_GPU_ONLY
Vector2f RandomGPU::get2D() {
  return {get1D(), get1D()};
}

F_CPU_GPU
void RandomGPU::init_cuRAND(uint64_t seed) {
  if constexpr (is_on_gpu()) {
    curand_init(seed, 0, 0, &state);
  } else {
    detail_::curand_init_Kernel<<<1, 1>>>(seed, 0, 0, &state);
  }
}

FLG_NAMESPACE_END