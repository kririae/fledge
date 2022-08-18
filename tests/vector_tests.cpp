#include <gtest/gtest.h>

#include <memory>

#include "aabb.hpp"
#include "accel.hpp"
#include "fwd.hpp"
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
}
