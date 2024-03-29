#ifndef __AABB_H__
#define __AABB_H__

#include "common/ray.h"
#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

class AABB {
public:
  F_CPU_GPU AABB() = default;
  F_CPU_GPU AABB(Vector3f t_min, Vector3f t_max) : m_min(t_min), m_max(t_max) {
    m_min = Min(t_min, t_max);
    m_max = Max(t_max, t_min);
  }
  F_CPU_GPU ~AABB() = default;

  F_CPU_GPU bool intersect(const Ray &ray, Float &t_min, Float &t_max) const {
    auto     inv_d  = ray.m_d.cwiseInverse();
    Vector3f vt1    = (m_min - ray.m_o) * (inv_d);
    Vector3f vt2    = (m_max - ray.m_o) * (inv_d);
    Vector3f vt_min = vt1.cwiseMin(vt2);
    Vector3f vt_max = vt1.cwiseMax(vt2);
    Float    l_min  = vt_min.maxCoeff();
    Float    l_max  = vt_max.minCoeff();
    if (l_min > l_max) std::swap(l_min, l_max);
    if (l_min < l_max && l_max >= 0) {
      t_min = l_min;
      t_max = l_max;
      return true;
    } else {
      return false;
    }
  }

  F_CPU_GPU bool intersect_pbrt(const Ray &ray, Float &t_min,
                                Float &t_max) const {
    Float t0 = 0, t1 = ray.m_tMax;
    for (int i = 0; i < 3; ++i) {
      // Update interval for _i_th bounding box slab
      Float inv_ray_dir = 1 / ray.m_d[i];
      Float t_near      = (m_min[i] - ray.m_o[i]) * inv_ray_dir;
      Float t_far       = (m_max[i] - ray.m_o[i]) * inv_ray_dir;

      // Update parametric interval from slab intersection $t$ values
      if (t_near > t_far) std::swap(t_near, t_far);

      // Update _tFar_ to ensure robust ray--bounds intersection
      t0 = t_near > t0 ? t_near : t0;
      t1 = t_far < t1 ? t_far : t1;
      if (t0 > t1) return false;
    }

    t_min = t0;
    t_max = t1;
    return true;
  }

  F_CPU_GPU bool inside(const Vector3f &p) const {
    return m_min[0] <= p[0] && m_min[1] <= p[1] && m_min[2] <= p[2] &&
           p[0] <= m_max[0] && p[1] <= m_max[1] && p[2] <= m_max[2];
  }

  F_CPU_GPU AABB merge(const AABB &other) {
    return AABB(m_min.cwiseMin(other.m_min), m_max.cwiseMax(other.m_max));
  }

  F_CPU_GPU void boundSphere(Vector3f &center, Float &radius) const {
    center = (m_min + m_max) / 2;
    radius = (m_max - center).norm();
  }

  F_CPU_GPU Float surfaceArea() const {
    Vector3f edges = m_max - m_min;
    return 2 * (edges.x() * edges.y() + edges.x() * edges.z() +
                edges.y() * edges.z());
  }

  F_CPU_GPU Vector3f center() const { return (m_min + m_max) / 2; }

  F_CPU_GPU bool operator==(const AABB &a) const {
    return m_min == a.m_min && m_max == a.m_max;
  }

  Vector3f m_min{Vector3f(0.0)}, m_max{Vector3f(0.0)};
};

FLG_NAMESPACE_END

#endif
