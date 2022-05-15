#include "accel.hpp"

#include "aabb.hpp"
#include "interaction.hpp"

SV_NAMESPACE_BEGIN

bool NaiveAccel::intersect(const Ray &ray, SInteraction &isect) const {
  Float        o_tMax = ray.m_tMax;
  Float        tMax   = o_tMax;
  SInteraction t_isect;
  for (auto &i : m_primitives) {
    if (i->intersect(ray, t_isect) && ray.m_tMax < tMax) {
      isect = t_isect;
      tMax  = ray.m_tMax;
    }
  }

  return tMax != o_tMax;
}

AABB NaiveAccel::getBound() const {
  AABB res;
  for (auto &primitive : m_primitives) res = res.merge(primitive->getBound());
  return res;
}

SV_NAMESPACE_END
