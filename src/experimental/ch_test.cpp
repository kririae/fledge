#include <memory_resource>

#include "common/vector.h"
#include "debug.hpp"
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

int main() {
  std::pmr::memory_resource *mem_resource = std::pmr::get_default_resource();
  Resource                   resource{mem_resource};
  TriangleMesh              *mesh =
      fledge::MakeTriangleMesh("assets/bun_zipper_res4.ply", resource);

  // Convert mesh to internal mesh
  InternalTriangleMesh *imesh =
      new InternalTriangleMesh{.nInd  = mesh->nInd,
                               .nVert = mesh->nVert,
                               .ind   = mesh->ind,
                               .p     = mesh->p,
                               .n     = mesh->n,
                               .uv    = mesh->uv};  // convert mesh
  ConvexHullBuilder  builder(imesh, mem_resource);
  ConvexHullInstance instance = builder.build();
  auto               _        = instance.toDCEL();

  auto *ch_imesh = instance.toITriangleMesh();
  // Inverse conversion
  TriangleMesh *ch_mesh = new TriangleMesh{.nInd  = ch_imesh->nInd,
                                           .nVert = ch_imesh->nVert,
                                           .ind   = ch_imesh->ind,
                                           .p     = ch_imesh->p,
                                           .n     = ch_imesh->n,
                                           .uv = ch_imesh->uv};  // convert mesh

  display(ch_mesh, "ch_mesh.exr");

  delete imesh;
  delete ch_mesh;
}