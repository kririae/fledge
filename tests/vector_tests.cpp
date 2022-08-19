#include <gtest/gtest.h>

#include <memory>

#include "aabb.hpp"
#include "accel.hpp"
#include "fwd.hpp"
#include "gtest/gtest.h"
#include "interaction.hpp"
#include "primitive.hpp"
#include "rng.hpp"
#include "shape.hpp"
#include "vector.hpp"

using namespace SmallVolNS;
inline Vector<Float, 3> MakeVector3f(Float x, Float y, Float z) {
  return {x, y, z};
}

TEST(Vector, Comprehensive) {
  Random rng;

  Vector<Float, 3> a{1, 2, 3};
  Vector<Float, 3> b{4, 5, 6};
  auto             c1 = a + b;
  auto             c2 = a * b;

  EXPECT_EQ(c1, MakeVector3f(5, 7, 9));
  EXPECT_EQ(c2, MakeVector3f(4, 10, 18));
  EXPECT_EQ(a * 2, MakeVector3f(2, 4, 6));
  EXPECT_EQ(2 * a, MakeVector3f(2, 4, 6));

  Vector<int, 3> d{7, 8, 9};
  bool res = std::is_same_v<decltype(d.cast<Float, 3>())::type, Float>;
  EXPECT_TRUE(res);

  EXPECT_EQ(Sum(a), 6);
  EXPECT_EQ(SquaredNorm(a), 14);
  EXPECT_NEAR(Norm(a), 3.74165738677394, 1e-5);
  EXPECT_EQ(Normalize(MakeVector3f(2, 0, 0)), MakeVector3f(1, 0, 0));
  EXPECT_EQ(Dot(a, b), 32);
  EXPECT_EQ(Cross(MakeVector3f(1, 0, 0), MakeVector3f(0, 1, 0)),
            MakeVector3f(0, 0, 1));
}
