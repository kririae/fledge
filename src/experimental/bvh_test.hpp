#ifndef __EXPERIMENTAL_BVH_TEST_HPP__
#define __EXPERIMENTAL_BVH_TEST_HPP__

#include "experimental/radix_bvh.hpp"
#define FMT_HEADER_ONLY  // use header-only fmt
#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/core.h>

#include <chrono>
#include <memory_resource>

#include "experimental/base_bvh.hpp"
#include "fledge.h"
#include "plymesh.hpp"
#include "resource.hpp"
#include "rng.hpp"
#include "shape.hpp"

FLG_NAMESPACE_BEGIN
namespace experimental {
struct TimeInfo {
  std::size_t ms;
  std::string desc;
};
struct Event {
  using clock = std::chrono::high_resolution_clock;
  Event(const std::string &desc) : m_start(clock::now()), m_desc(desc) {}
  TimeInfo end(bool print = false) {
    using namespace std::literals::chrono_literals;
    TimeInfo info{.ms   = static_cast<size_t>((clock::now() - m_start) / 1ms),
                  .desc = m_desc};
    if (print)
      fmt::print(fg(fmt::color::steel_blue), "[{}] takes {} ms\n", m_desc,
                 info.ms);
    return info;
  }

private:
  decltype(clock::now()) m_start;
  std::string            m_desc;
};

enum class EBVHType { EBVHBasic = 0, EBVHEmbree, EBVHRadix };

class BVHTester {
public:
  BVHTester(EBVHType type, std::pmr::memory_resource *mem_resource)
      : m_mem_resource(mem_resource), m_type(type), m_resource(mem_resource) {}
  virtual ~BVHTester() { delete m_imesh; }
  virtual bool loadMesh(const std::string &path) {
    Event         e_mesh_load("Mesh load time");
    TriangleMesh *mesh = fledge::MakeTriangleMesh(path, m_resource);
    e_mesh_load.end();

    // Convert mesh to internal mesh
    m_imesh = new InternalTriangleMesh{.nInd  = mesh->nInd,
                                       .nVert = mesh->nVert,
                                       .ind   = mesh->ind,
                                       .p     = mesh->p,
                                       .n     = mesh->n,
                                       .uv    = mesh->uv};  // convert mesh
    return true;
  }  // loadMesh
  virtual bool loadSphere() {
    Event         e_mesh_load("Mesh load time");
    TriangleMesh *mesh = fledge::MakeMeshedSphere(1000, 1000, 5, m_resource);
    e_mesh_load.end();

    // Convert mesh to internal mesh
    m_imesh = new InternalTriangleMesh{.nInd  = mesh->nInd,
                                       .nVert = mesh->nVert,
                                       .ind   = mesh->ind,
                                       .p     = mesh->p,
                                       .n     = mesh->n,
                                       .uv    = mesh->uv};  // convert mesh
    return true;
  }

  virtual bool build() {
    m_bvh_ref = m_resource.alloc<RefBVHBuilder>(m_imesh, m_mem_resource);
    switch (m_type) {
      case EBVHType::EBVHBasic:
        m_bvh_test = m_resource.alloc<BasicBVHBuilder>(m_imesh, m_mem_resource);
        break;
      case EBVHType::EBVHEmbree:
        m_bvh_test = m_resource.alloc<RefBVHBuilder>(m_imesh, m_mem_resource);
        break;
      case EBVHType::EBVHRadix:
        m_bvh_test = m_resource.alloc<RadixBVHBuilder>(m_imesh, m_mem_resource);
        break;
      default:
        TODO();
    }

    Event e_build("BVH build time");
    m_bvh_test->build();
    e_build.end(true);

    Event e_ref_build("Reference BVH build time");
    m_bvh_ref->build();
    e_ref_build.end(true);

    return true;
  }

  virtual bool correctness1(int N = 500000) {
    Event     e_c("BVH test time");
    RandomCPU rng{};
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
      bool inter1  = m_bvh_test->intersect(rayhit1);
      bool inter2  = m_bvh_ref->intersect(rayhit2);

      bool cond1 = inter1 == inter2, cond2, cond3;
      cond2 = cond3 = false;
      if (!cond1) {
        fmt::print("cond1: {} {} {} {}\n", rayhit1.ray_o.toString(),
                   rayhit1.ray_d.toString(), inter1, inter2);
        fmt::print("{} {} {}\n", rayhit2.tnear, rayhit2.tfar, i);
      }
      if (!cond1) return false;
      cond2 = abs(rayhit1.tfar - rayhit2.tfar) < 1e-3;
      cond3 = Same(rayhit1.hit_ng, rayhit2.hit_ng, 1e-3);
      if (!cond2) fmt::print("cond2: {} {}\n", rayhit1.tfar, rayhit2.tfar);
      if (!cond3)
        fmt::print("cond3: {} {} {}\n", rayhit1.hit_ng.toString(),
                   rayhit2.hit_ng.toString(), i);
      if (cond2 != true || cond3 != true) return false;
    }
    e_c.end(true);
    return true;
  }

  virtual bool correctness2() {
    BVHBuilderBase::BVHRayHit rayhit1{
        .ray_o = Vector3f{               0.345564,                   0.422557,0.259793                                          },
        .ray_d = Vector3f{0.524748325347900390625, 0.714325249195098876953125,
                          0.46301043033599853515625},
        .tnear = 0,
        .tfar  = std::numeric_limits<float>::max()
    };
    auto rayhit2 = rayhit1;

    bool inter1 = m_bvh_test->intersect(rayhit1);
    bool inter2 = m_bvh_ref->intersect(rayhit2);

    Vector3f point = rayhit2.hit_ng;
    fmt::print("{}\n", point.toString());
    Float min_distance = 1000000;
    for (int i = 0; i < m_imesh->nVert; ++i) {
      Vector3f vertex = m_imesh->p[i];
      min_distance    = std::min(min_distance, Norm(vertex - point));
    }

    assert(inter1 == true);
    assert(inter2 == true);
    fmt::print("test2 passed\n");
    return false;
  }

protected:
  std::pmr::memory_resource *m_mem_resource;

  EBVHType              m_type;
  Resource              m_resource;
  BVHBuilderBase       *m_bvh_test, *m_bvh_ref;
  InternalTriangleMesh *m_imesh;

private:
};
}  // namespace experimental
FLG_NAMESPACE_END

#endif
