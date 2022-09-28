#include <gtest/gtest.h>

#include "common/rng.h"
#include "fledge.h"
#include "gpu/gpu_rng.hpp"

using namespace fledge;

TEST(RNG, Basic) {
  RandomGPU *rng_gpu_ptr;
  cudaMallocManaged(&rng_gpu_ptr, sizeof(RandomGPU));
  new (rng_gpu_ptr) RandomGPU{};
  Random *rng;
  cudaMallocManaged(&rng, sizeof(Random));
  new (rng) Random{};
}