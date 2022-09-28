#ifndef __GPU_RNG_CUH__
#define __GPU_RNG_CUH__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <utility>

#include "common/rng.h"
#include "common/switcher.h"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

class RandomGPU : public Random {
public:
  explicit RandomGPU(uint64_t seed = 0) { init_cuRAND(seed); }
  ~RandomGPU() = default;

  F_GPU_ONLY
  Float get1D();
  F_GPU_ONLY
  Vector2f get2D();

private:
  F_CPU_GPU
  void init_cuRAND(uint64_t seed);

  curandState state;
};

FLG_NAMESPACE_END

#endif
