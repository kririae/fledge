#include <gtest/gtest.h>

#include <memory>

#include "common/aabb.h"
#include "accel.hpp"
#include "fledge.h"
#include "debug.hpp"
#include "interaction.hpp"
#include "primitive.hpp"
#include "rng.hpp"
#include "shape.hpp"
#include "common/vector.h"

TEST(Accel, NaiveBVHAccel) {
  using namespace fledge;
  Random rng;

  auto diffuse = std::make_shared<DiffuseMaterial>(Vector3f(1.0));

  std::vector<std::shared_ptr<Primitive>> p;

  // Init 100 sphere into the scene
  for (int i = 0; i < 200; ++i) {
    auto sphere = std::make_shared<Sphere>(
        Vector3f{rng.get1D(), rng.get1D(), rng.get1D()} * 20.0, 2);
    p.push_back(std::make_shared<ShapePrimitive>(sphere, diffuse));
  }

  NaiveAccel    acc(p);
  NaiveBVHAccel bacc(p);
  ASSERT_EQ(acc.getBound(), bacc.getBound());
  std::clog << "memory usage: " << bacc.getMemoryUsage() / 1024 << "kB"
            << std::endl;
  std::clog << "depth: " << bacc.getDepth() << std::endl;

  for (int i = 0; i < 10; ++i) {
    auto ray1 =
        Ray(Vector3f(0.0),
            Vector3f(rng.get1D(), rng.get1D(), rng.get1D()).stableNormalized());
    auto         ray2 = ray1;
    SInteraction isect1, isect2;

    bool res1 = acc.intersect(ray1, isect1);
    bool res2 = bacc.intersect(ray2, isect2);
    ASSERT_EQ(res1, res2);
    if (res1 && res2) {
      ASSERT_EQ(isect1.m_p, isect2.m_p);
      ASSERT_EQ(isect1.m_ng, isect2.m_ng);
      ASSERT_EQ(isect1.m_ns, isect2.m_ns);
      ASSERT_EQ(ray1.m_tMax, ray2.m_tMax);
    }
  }
}
