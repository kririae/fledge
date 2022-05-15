#ifndef __AABB_HPP__
#define __AABB_HPP__

#include "fwd.hpp"
#include "ray.hpp"

SV_NAMESPACE_BEGIN

class AABB {
public:
  AABB() = default;
  AABB(Vector3f t_min, Vector3f t_max) : m_min(t_min), m_max(t_max) {
    m_min = t_min.cwiseMin(t_max);
    m_max = t_max.cwiseMax(t_min);
  }
  ~AABB() = default;

  bool intersect(const Ray &ray, Float &t_min, Float &t_max) const {
    auto     inv_d  = ray.m_d.cwiseInverse();
    Vector3f vt1    = (m_min - ray.m_o).cwiseProduct(inv_d);
    Vector3f vt2    = (m_max - ray.m_o).cwiseProduct(inv_d);
    Vector3f vt_min = vt1.cwiseMin(vt2);
    Vector3f vt_max = vt1.cwiseMax(vt2);
    Float    l_min  = vt_min.maxCoeff();
    Float    l_max  = vt_max.minCoeff();
    if (l_min < l_max && l_max >= 0) {
      t_min = l_min;
      t_max = l_max;
      return true;
    } else {
      return false;
    }
  }

  bool intersect_pbrt(const Ray &ray, Float &t_min, Float &t_max) const {
    Float t0 = 0, t1 = ray.m_tMax;
    for (int i = 0; i < 3; ++i) {
      // Update interval for _i_th bounding box slab
      Float invRayDir = 1 / ray.m_d[i];
      Float tNear     = (m_min[i] - ray.m_o[i]) * invRayDir;
      Float tFar      = (m_max[i] - ray.m_o[i]) * invRayDir;

      // Update parametric interval from slab intersection $t$ values
      if (tNear > tFar) std::swap(tNear, tFar);

      // Update _tFar_ to ensure robust ray--bounds intersection
      t0 = tNear > t0 ? tNear : t0;
      t1 = tFar < t1 ? tFar : t1;
      if (t0 > t1) return false;
    }

    t_min = t0;
    t_max = t1;
    return true;
  }

  bool inside(const Vector3f &p) const {
    return m_min[0] <= p[0] && m_min[1] <= p[1] && m_min[2] <= p[2] &&
           p[0] <= m_max[0] && p[1] <= m_max[1] && p[2] <= m_max[2];
  }

  AABB merge(const AABB &other) {
    return AABB(m_min.cwiseMin(other.m_min), m_max.cwiseMax(other.m_max));
  }

  void boundSphere(Vector3f &center, Float &radius) const {
    center = (m_min + m_max) / 2;
    radius = (m_max - center).norm();
  }

private:
  Vector3f m_min{Vector3f::Zero()}, m_max{Vector3f::Zero()};
};

SV_NAMESPACE_END

#endif
