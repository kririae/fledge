#include <memory_resource>

#include "common/vector.h"
#include "debug.hpp"
#include "experimental/base_bvh.hpp"
#include "experimental/convex_hull.hpp"
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
  std::pmr::memory_resource *mem_resource = std::pmr::get_default_resource();
  Resource                   resource{mem_resource};
  TriangleMesh              *mesh_1 =
      fledge::MakeTriangleMesh("assets/watertight.ply", resource);
  TriangleMesh *mesh_2 = fledge::CloneTriangleMesh(mesh_1, resource);

  // Offset the mesh to perform intersection
  const Vector3f offset(1.9, 0.1, 0);
  for (int i = 0; i < mesh_2->nVert; ++i) mesh_2->p[i] += offset;

  InternalTriangleMesh imesh_1 = toITriangleMesh(mesh_1);
  InternalTriangleMesh imesh_2 = toITriangleMesh(mesh_2);

  ConvexHullInstance instance_1(&imesh_1, mem_resource);
  ConvexHullInstance instance_2(&imesh_2, mem_resource);

  instance_1.assumeConvex();
  instance_2.assumeConvex();

  // Mesh direction validation not passed
  for (auto &face : instance_1.m_faces) {
    Vector3f *p      = instance_1.m_mesh->p;
    Vector3f  a      = p[face.index[0]];
    Vector3f  b      = p[face.index[1]];
    Vector3f  c      = p[face.index[2]];
    Vector3f  center = (a + b + c) / 3;
    assert(!fledge::experimental::detail_::SameDirection(
        fledge::experimental::detail_::Normal(a, b, c), center));
  }

  fmt::print("{}\n", GJKIntersection(instance_1, instance_2));
}

int main() {
  std::pmr::memory_resource *mem_resource = std::pmr::get_default_resource();
  Resource                   resource{mem_resource};
  TriangleMesh              *mesh =
      fledge::MakeTriangleMesh("assets/bun_zipper_res4.ply", resource);

  // Convert mesh to internal mesh
  InternalTriangleMesh imesh = toITriangleMesh(mesh);
  ConvexHullBuilder    builder(&imesh, mem_resource);
  ConvexHullInstance   instance = builder.build();
  auto                 _        = instance.toDCEL();
  auto                *ch_imesh = instance.toITriangleMesh();
  TriangleMesh         ch_mesh  = toTriangleMesh(ch_imesh);
  //   display(&ch_mesh, "ch_mesh.exr");

  intersectTest();
}