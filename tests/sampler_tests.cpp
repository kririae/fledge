#include <gtest/gtest.h>

#include <memory>

#include "accel.hpp"
#include "common/aabb.h"
#include "common/sampler.h"
#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"
#include "interaction.hpp"
#include "primitive.hpp"
#include "rng.hpp"
#include "shape.hpp"

TEST(Sampler, GetPrimeList) {
  using namespace fledge;
  int *prime       = detail_::GetPrimeList();
  int  prime_ref[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
  for (int i = 0; i < sizeof(prime_ref) / sizeof(int); ++i)
    EXPECT_EQ(prime[i], prime_ref[i]);
}

TEST(Sampler, HaltonSampler) {
  using namespace fledge;
  std::vector<int> counter(10);
  int              SPP = 128;
  int              N   = 128 * SPP;
  HaltonSampler    sampler(SPP, Vector2d{5, 5});
  sampler.setPixel(Vector2d{1, 1});

  for (int spp = 0; spp < SPP; ++spp) {
    for (int i = 0; i < 128; ++i) {
      Float sampled = sampler.get1D();
      int   s       = std::min(int(sampled * 10), 9);
      counter[s] += 1;
    }

    sampler.reset();
  }

  for (int i = 0; i < 10; ++i) {
    EXPECT_NEAR((Float)counter[i] / N, 0.1, 1e-2);
  }
}
