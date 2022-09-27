#include <gtest/gtest.h>

#include <memory>

#include "accel.hpp"
#include "common/aabb.h"
#include "common/math_utils.h"
#include "common/vector.h"
#include "debug.hpp"
#include "external/embree/eprimitive.hpp"
#include "fledge.h"
#include "interaction.hpp"
#include "plymesh.hpp"
#include "primitive.hpp"
#include "rng.hpp"
#include "shape.hpp"

TEST(Triangle, ComprehensiveTestWithEmbree) {
  using namespace fledge;
}

TEST(Triangle, Watertight) {
  // Copied from PBRT-v3
  using namespace fledge;

  RandomCPU        rng(114514);
  Resource      resource;
  TriangleMesh *mesh = MakeMeshedSphere(16, 16, 1, resource);

  auto mesh_primitive = MeshPrimitive(mesh, resource);

  for (int i = 0; i < 1000; ++i) {
    Vector2f u = rng.get2D();
    Vector3f p = Vector3f(0.0) + 0.5 * UniformSampleSphere(u);

    u = rng.get2D();
    Ray r(p, UniformSampleSphere(u));

    SInteraction isect1;
    EXPECT_TRUE(mesh_primitive.intersect(r, isect1));

    // // Might be too strict..
    // Vector3f vert = vertices[int(rng.get1D() * (vertices.size() - 1))];
    // r             = Ray(p, (vert - p).normalized());
    // SInteraction isect2;
    // EXPECT_TRUE(mesh_primitive.intersect(r, isect2));
  }
}

TEST(Sphere, intersect) {
  using namespace fledge;

  // Draw a sphere at zero
  auto sphere = Sphere(Vector3f(0.0), 1.0);

  Ray ray1(Vector3f(1.1, 0.0, 0.0), Vector3f(1.0, 0.0, 0.0));

  Float        tHit = 2.3;
  SInteraction isect;
  bool         result1 = sphere.intersect(ray1, tHit, isect);
  EXPECT_FALSE(result1);

  Ray  ray2(Vector3f(1.0, 0.0, 0.0), Vector3f(1.0, 0.0, 0.0));
  bool result2 = sphere.intersect(ray2, tHit, isect);
  EXPECT_TRUE(result2);
  EXPECT_NEAR(tHit, 0, 1e-4);

  Ray  ray3(Vector3f(0.6, 0.0, 0.0), Vector3f(1.0, 0.0, 0.0));
  bool result3 = sphere.intersect(ray3, tHit, isect);
  EXPECT_TRUE(result3);
  EXPECT_NEAR(tHit, 0.4, 1e-4);
  EXPECT_NEAR(isect.m_p.x(), 1.0, 1e-4);
  EXPECT_NEAR(isect.m_ng.x(), 1.0, 1e-4);

  Ray  ray4(Vector3f(-1.1, 0.0, 0.0), Vector3f(1.0, 0.0, 0.0));
  bool result4 = sphere.intersect(ray4, tHit, isect);
  EXPECT_TRUE(result4);
  EXPECT_NEAR(tHit, 0.1, 1e-4);
  EXPECT_NEAR(isect.m_p.x(), -1.0, 1e-4);
  EXPECT_NEAR(isect.m_ng.x(), -1.0, 1e-4);
}