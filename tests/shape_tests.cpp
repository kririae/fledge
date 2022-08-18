#include <gtest/gtest.h>

#include <memory>

#include "aabb.hpp"
#include "accel.hpp"
#include "embree/eprimitive.hpp"
#include "fwd.hpp"
#include "interaction.hpp"
#include "plymesh.hpp"
#include "primitive.hpp"
#include "rng.hpp"
#include "shape.hpp"
#include "utils.hpp"
#include "vector.hpp"

TEST(Triangle, ComprehensiveTestWithEmbree) {
  using namespace SmallVolNS;
}

TEST(Triangle, Watertight) {
  // Copied from PBRT-v3
  using namespace SmallVolNS;

  Random rng(114514);
  int    n_theta = 32, n_phi = 32;

  int  n_vert = n_theta * n_phi;
  auto mesh   = std::make_shared<TriangleMesh>();
  mesh->nVert = n_vert;
  std::vector<Vector3f> vertices;
  for (int t = 0; t < n_theta; ++t) {
    Float theta     = PI * (Float)t / (Float)(n_theta - 1);
    Float cos_theta = std::cos(theta);
    Float sin_theta = std::sin(theta);
    for (int p = 0; p < n_phi; ++p) {
      Float phi    = 2 * PI * (Float)p / (Float)(n_phi - 1);
      Float radius = 1;
      // Make sure all of the top and bottom vertices are coincident.
      if (t == 0)
        vertices.push_back(Vector3f(0, 0, radius));
      else if (t == n_theta - 1)
        vertices.push_back(Vector3f(0, 0, -radius));
      else if (p == n_phi - 1)
        // Close it up exactly at the end
        vertices.push_back(vertices[vertices.size() - (n_phi - 1)]);
      else {
        // radius += 5 * rng.get1D();
        vertices.push_back(Vector3f(0, 0, 0) +
                           radius *
                               SphericalDirection(sin_theta, cos_theta, phi));
      }
    }
  }

  std::vector<int> indices;
  // fan at top
  auto offset = [n_phi](int t, int p) { return t * n_phi + p; };
  for (int p = 0; p < n_phi - 1; ++p) {
    indices.push_back(offset(0, 0));
    indices.push_back(offset(1, p));
    indices.push_back(offset(1, p + 1));
  }

  // quads in the middle rows
  for (int t = 1; t < n_theta - 2; ++t) {
    for (int p = 0; p < n_phi - 1; ++p) {
      indices.push_back(offset(t, p));
      indices.push_back(offset(t + 1, p));
      indices.push_back(offset(t + 1, p + 1));

      indices.push_back(offset(t, p));
      indices.push_back(offset(t + 1, p + 1));
      indices.push_back(offset(t, p + 1));
    }
  }

  // fan at bottom
  for (int p = 0; p < n_phi - 1; ++p) {
    indices.push_back(offset(n_theta - 1, 0));
    indices.push_back(offset(n_theta - 2, p));
    indices.push_back(offset(n_theta - 2, p + 1));
  }

  mesh->nInd = indices.size();
  mesh->p    = std::make_unique<Vector3f[]>(mesh->nVert);
  mesh->ind  = std::make_unique<int[]>(mesh->nInd);
  for (int i = 0; i < mesh->nVert; ++i) mesh->p[i] = vertices[i];
  for (int i = 0; i < mesh->nInd; ++i) mesh->ind[i] = indices[i];

  auto mesh_primitive = MeshPrimitive(mesh);

  for (int i = 0; i < 1000; ++i) {
    Vector2f u = rng.get2D();
    Vector3f p = Vector3f::Zero() + 0.5 * UniformSampleSphere(u);

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
