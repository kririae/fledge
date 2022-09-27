#ifndef __EXPERIMENTAL_BASE_BVH_HPP__
#define __EXPERIMENTAL_BASE_BVH_HPP__

#include <embree3/rtcore.h>
#include <embree3/rtcore_geometry.h>
#include <oneapi/tbb/cache_aligned_allocator.h>

#include <cstddef>
#include <limits>
#include <memory_resource>

#include "common/aabb.h"
#include "common/vector.h"
#include "external/embree/embree.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN
namespace experimental {
namespace detail_ {
inline bool TriangleIntersect(const Vector3f &a, const Vector3f &b,
                              const Vector3f &c, const Vector3f &ray_o,
                              const Vector3f &ray_d, float *thit,
                              Vector3f *ng) {
  // Naive implementation
  Vector3f e1      = b - a;
  Vector3f e2      = c - a;
  Vector3f p       = Cross(ray_d, e2);
  Float    d       = e1.dot(p);
  float    inv_det = 1 / d;
  if (d == 0) return false;

  Vector3f t = ray_o - a;
  Float    u = t.dot(p) * inv_det;
  if (u < 0 || u > 1) return false;

  Vector3f q = t.cross(e1);
  Float    v = ray_d.dot(q) * inv_det;
  if (v < 0 || u + v > 1) return false;

  Float t_near = e2.dot(q) * inv_det;
  if (t_near <= 0) return false;

  if (thit != nullptr) *thit = t_near;
  if (ng != nullptr) *ng = Normalize(Cross(e1, e2));
  return true;
}

inline bool BoundIntersect(const Vector3f &lower, const Vector3f &upper,
                           const Vector3f &ray_o, const Vector3f &ray_d,
                           float &tnear, float &tfar) {
  Float t0 = 0, t1 = std::numeric_limits<float>::max();
  for (int i = 0; i < 3; ++i) {
    Float inv_ray_dir = 1 / ray_d[i];
    Float t_near      = (lower[i] - ray_o[i]) * inv_ray_dir;
    Float t_far       = (upper[i] - ray_o[i]) * inv_ray_dir;
    if (t_near > t_far) std::swap(t_near, t_far);
    t0 = t_near > t0 ? t_near : t0;
    t1 = t_far < t1 ? t_far : t1;
    if (t0 > t1) return false;
  }  // for
  tnear = t0, tfar = t1;
  return true;
}
}  // namespace detail_
// Some classes are borrowed from the original project
// 1. Math primitives, e.g., Vector3f
struct InternalTriangleMesh {
  int       nInd, nVert;
  int      *ind;   /* nInd = nVert * 3 */
  Vector3f *p, *n; /* nVert */
  Vector2f *uv;    /* nVert */
};                 // InternalTriangleMesh
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
    Vector3f lower{0}, upper{0};
    void     merge(const BVHBound &other) {
          // ??? format error
      lower = Min(lower, other.lower);
      upper = Max(upper, other.upper);
    }  // void merge()
    bool intersect(const Vector3f &ray_o, const Vector3f &ray_d, float &tnear,
                   float &tfar) {
      return detail_::BoundIntersect(lower, upper, ray_o, ray_d, tnear, tfar);
    }
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
  BasicBVHNode *recursiveBuilder(Triangle *triangles, std::size_t n_triangles,
                                 int depth);
  bool          recursiveIntersect(BasicBVHNode *node, BVHRayHit &rayhit);
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
