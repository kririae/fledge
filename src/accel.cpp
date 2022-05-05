#include "accel.hpp"

SV_NAMESPACE_BEGIN

bool NaiveAccel::intersect(const Ray &ray, SInteraction &isect) const {
  Float        tMax = INF;
  SInteraction t_isect;
  for (auto &i : m_primitives) {
    if (i->intersect(ray, t_isect) && ray.m_tMax < tMax) {
      isect = t_isect;
      tMax  = ray.m_tMax;
    }
  }

  return tMax != INF;
}

SV_NAMESPACE_END
