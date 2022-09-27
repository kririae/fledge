#include <gtest/gtest.h>

#include "distribution.hpp"
#include "fledge.h"
#include "debug.hpp"
#include "rng.hpp"
#include "common/vector.h"

using namespace fledge;

TEST(Distribution, Distribution1DNaiveSampleDiscrete) {
  Dist1D d({0, 1, 2, 3});
  EXPECT_EQ(d.size(), 4);
  RandomCPU rng(114514);
  int    N = 10000;

  std::vector<int> d_vec(4, 0);
  for (int i = 0; i < N; ++i) {
    Float pdf;
    int   res = d.sampleD(rng.get1D(), pdf);
    EXPECT_LT(res, 4);
    d_vec[res]++;
  }

  Float err = 1e-2;
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(d_vec[i] / N, i / 6, err);
}

TEST(Distribution, Distribution2DNaiveSample) {
  std::vector<Float> data{1, 2, 3, 4, /**/ 5, 6, 7, 8, /**/ 9, 10, 11, 12};

  int              nu = 4, nv = 3;
  Dist2D           d(data.data(), nu, nv);
  RandomCPU           rng(114514);
  int              N = 10000;
  std::vector<int> d_vec(data.size(), 0);
  Float            err = 1e-1;
  for (int i = 0; i < N; ++i) {
    Float    pdf;
    Vector2f res   = d.sampleC(rng.get2D(), pdf);
    int      index = res.x() * nu + res.y();
    // EXPECT_NEAR(pdf, index / 78, err);
    EXPECT_LT(res.x(), nv);
    EXPECT_LT(res.y(), nu);
    d_vec[res.x() * nu + res.y()]++;
  }
  for (int i = 0; i < data.size(); ++i) EXPECT_NEAR(d_vec[i] / N, i / 78, err);
}

TEST(Distribution, Distribution1DSingle) {
  RandomCPU rng(114514);
  auto   dist = std::make_shared<Dist1D>(std::vector<Float>(1, 1));
  int    off;
  Float  pdf;
  auto   res = dist->sampleC(rng.get1D(), pdf, off);
  EXPECT_EQ(pdf, 1);
}

TEST(Distribution, Distribution2DSingle) {
  RandomCPU rng(114514);
  auto   dist = std::make_shared<Dist2D>(std::vector<Float>(1, 1).data(), 1, 1);
  Float  pdf;
  auto   res = dist->sampleC(rng.get2D(), pdf);
  EXPECT_EQ(pdf, 1);
}