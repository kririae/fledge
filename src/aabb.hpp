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

  bool intersect(const Ray &ray, Float &t_min, Float &t_max) {
    Vector3f vt_min = (m_min - ray.m_o).cwiseProduct(ray.m_d.cwiseInverse());
    Vector3f vt_max = (m_max - ray.m_o).cwiseProduct(ray.m_d.cwiseInverse());
    Float    l_min  = vt_min.maxCoeff();
    Float    l_max  = vt_max.maxCoeff();
    if (l_min >= 0 && l_min < l_max) {
      t_min = vt_min.maxCoeff();
      t_max = vt_max.minCoeff();
      return true;
    } else {
      return false;
    }
  }

private:
  Vector3f m_min, m_max;
};

SV_NAMESPACE_END

#endif
