#include <gtest/gtest.h>

#include <memory>

#include "accel.hpp"
#include "common/aabb.h"
#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"
#include "interaction.hpp"
#include "primitive.hpp"
#include "rng.hpp"
#include "shape.hpp"

TEST(Accel, NaiveBVHAccel) {
  using namespace fledge;
  Random   rng;
  Resource resource;

  auto diffuse = resource.alloc<DiffuseMaterial>(Vector3f(1.0));

  std::vector<Primitive *> p;

  // Init 100 sphere into the scene
  for (int i = 0; i < 200; ++i) {
    auto sphere = resource.alloc<Sphere>(
        Vector3f{rng.get1D(), rng.get1D(), rng.get1D()} * 20.0, 2);
    p.push_back(resource.alloc<ShapePrimitive>(sphere, diffuse));
  }

  NaiveAccel    acc(p);
  NaiveBVHAccel bacc(p, resource);
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
