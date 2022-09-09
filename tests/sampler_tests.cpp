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
  int              N = 40960;
  HaltonSampler    sampler(32, 114514);
  sampler.reset();
  for (int i = 0; i < N; ++i) {
    int s = std::min(int(sampler.get1D() * 10), 9);
    counter[s] += 1;
  }

  for (int i = 0; i < 10; ++i) {
    EXPECT_NEAR((Float)counter[i] / N, 0.1, 1e-2);
  }
}
