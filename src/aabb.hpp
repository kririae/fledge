#ifndef __AABB_HPP__
#define __AABB_HPP__

#include "fwd.hpp"
#include "ray.hpp"

SV_NAMESPACE_BEGIN

class AABB {
public:
  AABB() = default;
  AABB(Vector3f t_min, Vector3f t_max) : m_min(t_min), m_max(t_max) {}
  ~AABB() = default;

  bool intersect(const Ray &ray, Float &t_min, Float &t_max) const {
    auto     inv_d  = ray.m_d.cwiseInverse();
    Vector3f vt1    = (m_min - ray.m_o).cwiseProduct(inv_d);
    Vector3f vt2    = (m_max - ray.m_o).cwiseProduct(inv_d);
    Vector3f vt_min = vt1.cwiseMin(vt2);
    Vector3f vt_max = vt1.cwiseMax(vt2);
    Float    l_min  = vt_min.maxCoeff();
    Float    l_max  = vt_max.minCoeff();
    if (l_min >= 0 && l_min < l_max) {
      t_min      = l_min;
      t_max      = l_max;
      ray.m_tMax = t_max;
      return true;
    } else {
      return false;
    }
  }

  bool inside(const Vector3f &p) const {
    return m_min[0] <= p[0] && m_min[1] <= p[1] && m_min[2] <= p[2] &&
           p[0] <= m_max[0] && p[1] <= m_max[1] && p[2] <= m_max[2];
  }

private:
  Vector3f m_min, m_max;
};

SV_NAMESPACE_END

#endif
