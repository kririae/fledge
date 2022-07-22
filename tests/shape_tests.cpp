#include <gtest/gtest.h>

#include <memory>

#include "aabb.hpp"
#include "accel.hpp"
#include "fwd.hpp"
#include "interaction.hpp"
#include "plymesh.hpp"
#include "primitive.hpp"
#include "rng.hpp"
#include "shape.hpp"

TEST(Shape, make_TriangleMesh) {
  using namespace SmallVolNS;
  auto mesh = make_TriangleMesh("../../assets/bun_zipper.ply");
  printf("%d\n", mesh->nInd);
}
