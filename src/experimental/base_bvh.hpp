#ifndef __EXPERIMENTAL_BASE_BVH_HPP__
#define __EXPERIMENTAL_BASE_BVH_HPP__

#include <embree3/rtcore.h>
#include <embree3/rtcore_geometry.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <sys/cdefs.h>

#include <cstddef>
#include <limits>
#include <memory_resource>
#include <stdexcept>

#include "common/aabb.h"
#include "common/math_utils.h"
#include "common/vector.h"
#include "experimental/intersector.hpp"
#include "external/embree/embree.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN
namespace experimental {
// Some classes are borrowed from the original project
// 1. Math primitives, e.g., Vector3f
// template <bool isConvex = false>
struct InternalTriangleMesh {
  int       nInd, nVert;
  int      *ind;   /* nInd = nVert * 3 */
  Vector3f *p, *n; /* nVert */
  Vector2f *uv;    /* nVert */
};                 // InternalTriangleMesh

enum class EBVHPartitionMethod {
  ENone = 0,
  ESAH  = 1,
};

class BVHBuilderBase {
public:
  // TODO: to be extended to SIMD packet
  struct alignas(16) BVHRayHit {
    Vector3f ray_o{0};                                /* init */
    Vector3f ray_d{0};                                /* init */
    float    tnear{0};                                /* init */
    float    tfar{std::numeric_limits<float>::max()}; /* init, modified */

    Vector3f hit_ng{0};  /* modified */
    Vector3f hit_ns{0};  /* modified */
    bool     hit{false}; /* modified */
  };

  // TODO: to be extended to SIMD
  struct alignas(16) BVHBound {
    Vector3f lower{std::numeric_limits<Float>::max()},
        upper{-std::numeric_limits<Float>::max()};
    void merge(const BVHBound &other) {
      // ??? format error
      lower = Min(lower, other.lower);
      upper = Max(upper, other.upper);
    }  // void merge()
    Float surfaceArea() const {
      // assert(lower.x() != std::numeric_limits<Float>::max());
      Vector3f edges = upper - lower;
      return 2 * (edges.x() * edges.y() + edges.x() * edges.z() +
                  edges.y() * edges.z());
    }  // Float surfaceArea()
    Float volume() const {
      Vector3f edges = upper - lower;
      return edges.x() * edges.y() * edges.z();
    }
    bool intersect(const Vector3f &ray_o, const Vector3f &ray_d, float &tnear,
                   float &tfar) const {
      assert(lower.x() != std::numeric_limits<Float>::max());
      return detail_::BoundIntersect(lower, upper, ray_o, ray_d, tnear, tfar);
    }
    bool isValid() { return lower.x() != std::numeric_limits<Float>::max(); }
  };

  struct alignas(16) Triangle {
    int                  *m_v;    /* init */
    InternalTriangleMesh *m_mesh; /* init */
    inline Vector3f       a() const { return m_mesh->p[*m_v]; }
    inline Vector3f       b() const { return m_mesh->p[*(m_v + 1)]; }
    inline Vector3f       c() const { return m_mesh->p[*(m_v + 2)]; }
    inline float    centerX3() const { return a().x() + b().x() + c().x(); }
    inline float    centerY3() const { return a().y() + b().y() + c().y(); }
    inline float    centerZ3() const { return a().z() + b().z() + c().z(); }
    inline Vector3f center3() const { return a() + b() + c(); }
    inline Vector3f center() const { return center3() / 3; }
    BVHBound        getBound() const {
             // ??? format error
      const auto &p0 = a();
      const auto &p1 = b();
      const auto &p2 = c();
      BVHBound    bound;
      // Manually init the boundary to save computation resource
      bound.lower = Min(p0, p1);
      bound.lower = Min(bound.lower, p2);
      bound.upper = Max(p0, p1);
      bound.upper = Max(bound.upper, p2);
      return bound;
    }  // BVHBound getBound()
  };

  BVHBuilderBase(InternalTriangleMesh      *mesh,
                 std::pmr::memory_resource *mem_resource)
      : m_mesh(mesh),
        m_has_normal(mesh->n != nullptr),
        m_mem_resource(mem_resource) {}
  virtual ~BVHBuilderBase()                     = default;
  virtual void     build()                      = 0;
  virtual BVHBound getBound() const             = 0;
  virtual bool     intersect(BVHRayHit &rayhit) = 0;
  // virtual BuilderInfo getBuilderInfo() = delete;
  virtual std::pmr::memory_resource *getMemoryResource() {
    return m_mem_resource;
  }

protected:
  InternalTriangleMesh *m_mesh;
  bool                  m_has_normal; /* info about the mesh */

  std::pmr::memory_resource *m_mem_resource{
      std::pmr::get_default_resource()}; /* For memory management */
};

class BasicBVHBuilder : public BVHBuilderBase {
public:
  struct BasicBVHNode {
    BVHBound      bound;
    BasicBVHNode *left{nullptr}, *right{nullptr};
    Triangle     *triangles;
    std::size_t   n_triangles;
  };

  BasicBVHBuilder(InternalTriangleMesh      *mesh,
                  std::pmr::memory_resource *mem_resource)
      : BVHBuilderBase(mesh, mem_resource),
        m_upstream(m_mem_resource),
        m_resource(&m_upstream) {}
  ~BasicBVHBuilder() override {}

  /**
   * @brief build() is expected to initialize m_root using the memory resource
   * provided in m_resource
   */
  void     build() override;
  BVHBound getBound() const override { return m_root->bound; }
  bool     intersect(BVHRayHit &rayhit) override;

protected:
  BasicBVHNode *m_root;
  BasicBVHNode *recursiveBuilder(
      Triangle *triangles, std::size_t n_triangles, int depth,
      EBVHPartitionMethod partition_method = EBVHPartitionMethod::ENone);
  bool recursiveIntersect(BasicBVHNode *node, BVHRayHit &rayhit,
                          std::size_t &n_intersect);
  /**
   * The specific memory resource manager inherited from BaseClass.
   */
  tbb::cache_aligned_resource          m_upstream{m_mem_resource};
  std::pmr::synchronized_pool_resource m_resource{&m_upstream};
};

/**
 * @brief The reference BVH builder implemented in embree
 * @note Although a more elaborate implementation is provided, we'll use its
 * simplest API in this class
 */
class RefBVHBuilder : public BVHBuilderBase {
public:
  RefBVHBuilder(InternalTriangleMesh      *mesh,
                std::pmr::memory_resource *mem_resource);

  /**
   * The three basic functions to be translated into embree API
   */
  void     build() override;
  BVHBound getBound() const override;
  bool     intersect(BVHRayHit &rayhit) override;

private:
  /**
   * Here are the properties used by embree3
   */
  RTCDevice    m_device;
  RTCScene     m_scene;
  RTCGeometry  m_geom;
  unsigned int m_geomID;
};
}  // namespace experimental
FLG_NAMESPACE_END

#endif
