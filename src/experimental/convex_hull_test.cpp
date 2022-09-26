#include "experimental/convex_hull.hpp"

#include <oneapi/tbb/scalable_allocator.h>
#include <pstl/glue_execution_defs.h>

#include <algorithm>
#include <execution>
#include <memory_resource>

#include "common/vector.h"
#include "debug.hpp"
#include "experimental/base_bvh.hpp"
#include "film.hpp"
#include "fledge.h"
#include "light.hpp"
#include "plymesh.hpp"
#include "primitive.hpp"
#include "render.hpp"
#include "rng.hpp"
#include "scene.hpp"
#include "shape.hpp"
#include "spec/embree/eprimitive.hpp"
#include "texture.hpp"
#include "volume.hpp"

using namespace fledge;
using namespace fledge::experimental;

static InternalTriangleMesh toITriangleMesh(const TriangleMesh *mesh) {
  InternalTriangleMesh imesh{.nInd  = mesh->nInd,
                             .nVert = mesh->nVert,
                             .ind   = mesh->ind,
                             .p     = mesh->p,
                             .n     = mesh->n,
                             .uv    = mesh->uv};  // convert mesh
  return imesh;
}

static TriangleMesh toTriangleMesh(const InternalTriangleMesh *imesh) {
  TriangleMesh mesh{.nInd  = imesh->nInd,
                    .nVert = imesh->nVert,
                    .ind   = imesh->ind,
                    .p     = imesh->p,
                    .n     = imesh->n,
                    .uv    = imesh->uv};  // convert imesh
  return mesh;
}

static void display(TriangleMesh *mesh, const std::string &name) {
  Scene scene;

  auto mat = scene.m_resource.alloc<DiffuseMaterial>(Vector3f(1.0));
  scene.m_primitives.clear();
  scene.m_primitives.push_back(
      scene.m_resource.alloc<EmbreeMeshPrimitive>(mesh, scene.m_resource, mat));

  auto env_texture =
      std::make_shared<ImageTexture>("assets/venice_sunset_4k.exr");
  scene.m_light.clear();
  scene.m_infLight.clear();
  scene.m_light.push_back(
      scene.m_resource.alloc<InfiniteAreaLight>(env_texture));
  scene.m_infLight.push_back(scene.m_light[scene.m_light.size() - 1]);

  scene.m_resX            = 1280;
  scene.m_resY            = 720;
  scene.m_SPP             = 32;
  scene.m_maxDepth        = 16;
  scene.m_FoV             = 30;  // y axis
  scene.m_up              = Vector3f(0, 1, 0);
  scene.m_origin          = Vector3f(0, 0.1, 0.3);
  scene.m_target          = Vector3f(0, 0.1, 0);
  scene.m_volume          = nullptr;
  scene.m_base_dir        = std::filesystem::path(".");
  scene.m_integrator_type = EIntegratorType::EPathIntegrator;

  scene.init();

  auto  scene_aabb = scene.getBound();
  Float radius;
  scene_aabb.boundSphere(scene.m_target, radius);
  scene.m_origin = Normalize(Vector3f{0, 0.1, 0.3}) * radius * 3;

  scene.init();  // reinit;

  Render render(&scene);
  render.init();
  render.preprocess();
  render.render();

  render.saveImage(name);
}

static void intersectTest() {
  Random rng;

  std::pmr::memory_resource *mem_resource = std::pmr::get_default_resource();
  Resource                   resource{mem_resource};
  TriangleMesh              *mesh_1 =
      fledge::MakeTriangleMesh("assets/sphere.ply", resource);
  TriangleMesh *mesh_2 = fledge::CloneTriangleMesh(mesh_1, resource);

  InternalTriangleMesh imesh_1 = toITriangleMesh(mesh_1);
  InternalTriangleMesh imesh_2 = toITriangleMesh(mesh_2);

  ConvexHullInstance instance_1(&imesh_1, mem_resource);
  ConvexHullInstance instance_2(&imesh_2, mem_resource);

  instance_1.assumeConvex();
  instance_2.assumeConvex();

  assert(instance_1.verifyOrientation());
  assert(instance_2.verifyOrientation());

  auto i2_pos_array =
      std::span{imesh_2.p, static_cast<std::size_t>(imesh_2.nVert)};

  for (int i = 0; i < 100; ++i) {
    float    distance = rng.get1D() * 0.4;
    Vector3f direction =
        Normalize(Vector3f{rng.get1D(), rng.get1D(), rng.get1D()});
    Vector3f offset = distance * direction;

    // Transform
    std::transform(std::execution::par, i2_pos_array.begin(),
                   i2_pos_array.end(), i2_pos_array.begin(),
                   [=](const Vector3f &x) -> Vector3f { return x + offset; });

    bool intersect_result = GJKIntersection(instance_1, instance_2);
    if (distance < 0.199) assert(intersect_result);  // some extent of tolerance

    // Inverse transform
    std::transform(std::execution::par, i2_pos_array.begin(),
                   i2_pos_array.end(), i2_pos_array.begin(),
                   [=](const Vector3f &x) -> Vector3f { return x - offset; });
  }
}

int main() {
  auto                       upstream = std::pmr::monotonic_buffer_resource{};
  std::pmr::memory_resource *mem_resource = &upstream;
  Resource                   resource{mem_resource};
  TriangleMesh              *mesh =
      fledge::MakeTriangleMesh("assets/bun_zipper_res4.ply", resource);
#if 0
  TriangleMesh        *mesh_  = CloneTriangleMesh(mesh, resource);
  InternalTriangleMesh imesh_ = toITriangleMesh(mesh_);
  ConvexHullInstance   instance_(&imesh_, mem_resource);
  instance_.assumeConvex();
  assert(instance_.verifyOrientation());
#endif

  // Convert mesh to internal mesh
  InternalTriangleMesh imesh = toITriangleMesh(mesh);
  ConvexHullBuilder    builder(&imesh, mem_resource);
  ConvexHullInstance   instance = builder.build();
  auto                 _        = instance.toDCEL();
  auto                *ch_imesh = instance.toITriangleMesh();
  TriangleMesh         ch_mesh  = toTriangleMesh(ch_imesh);
  assert(instance.verifyOrientation());

#if 0
  display(&ch_mesh, "ch_mesh.exr");
#endif

  intersectTest();
  upstream.release();  // release all memory used
}