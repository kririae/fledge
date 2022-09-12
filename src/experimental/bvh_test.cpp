#include "bvh_test.hpp"

#include <iostream>
#include <limits>
#include <memory_resource>

#include "base_bvh.hpp"
#include "plymesh.hpp"
#include "rng.hpp"
#include "shape.hpp"

using namespace fledge;
using namespace fledge::experimental;
using namespace std::literals::chrono_literals;

static void test_BasicBVHBuilder() {
  std::pmr::memory_resource *mem_resource = std::pmr::get_default_resource();
  Resource                   resource(mem_resource);

  Event         e_mesh_load("Mesh load time");
  TriangleMesh *mesh = MakeMeshedSphere(1000, 1000, 5, resource);
  e_mesh_load.end();

  InternalTriangleMesh *imesh =
      new InternalTriangleMesh{.nInd  = mesh->nInd,
                               .nVert = mesh->nVert,
                               .ind   = mesh->ind,
                               .p     = mesh->p,
                               .n     = mesh->n,
                               .uv    = mesh->uv};  // convert mesh
  BVHBuilderBase *builder =
      resource.alloc<BasicBVHBuilder>(imesh, mem_resource);
  BVHBuilderBase *ref_builder =
      resource.alloc<RefBVHBuilder>(imesh, mem_resource);

  Event e_build("BVH build time");
  builder->build();
  e_build.end();

  Event e_ref_build("Reference BVH build time");
  ref_builder->build();
  e_ref_build.end();

  fmt::print("bounds: {}, {}\n", builder->getBound().lower.toString().c_str(),
             builder->getBound().upper.toString().c_str());

  constexpr int N = 1000;
  Random        rng;
  for (int i = 0; i < N; ++i) {
    BVHBuilderBase::BVHRayHit rayhit1{
        .ray_o = Vector3f{rng.get1D(), rng.get1D(), rng.get1D()}
          / 2,
        .ray_d = Normalize(Vector3f{rng.get1D(), rng.get1D(), rng.get1D()}
          ),
        .tnear = 0,
        .tfar  = std::numeric_limits<float>::max()
    };
    auto rayhit2 = rayhit1;
    bool inter1  = builder->intersect(rayhit1);
    bool inter2  = ref_builder->intersect(rayhit2);
    assert(inter1 == inter2);
    assert(abs(rayhit1.tfar - rayhit2.tfar) < 1e-4);
    assert(Same(rayhit1.hit_ng, rayhit2.hit_ng, 1e-4));
  }

  delete imesh;
}

int main() {
  test_BasicBVHBuilder();
}