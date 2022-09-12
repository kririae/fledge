#include "bvh.hpp"

#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/core.h>

#include <chrono>
#include <iostream>
#include <memory_resource>

#include "plymesh.hpp"
#include "shape.hpp"

using namespace fledge;
using namespace fledge::experimental;
using namespace std::literals::chrono_literals;

struct Event {
  using clock = std::chrono::high_resolution_clock;
  Event(const std::string &desc) : m_start(clock::now()), m_desc(desc) {}
  void end() {
    fmt::print("[{}] takes {} ms\n", m_desc, (clock::now() - m_start) / 1ms);
  }

private:
  decltype(clock::now()) m_start;
  std::string            m_desc;
};

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

  Event e_build("BVH build time");
  builder->build();
  e_build.end();

  fmt::print("bounds: {}, {}", builder->getBound().lower.toString().c_str(),
             builder->getBound().upper.toString().c_str());

  delete imesh;
}

int main() {
  test_BasicBVHBuilder();
}