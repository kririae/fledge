#include "fwd.hpp"
#include "integrator.hpp"
#include "light.hpp"
#include "primitive.hpp"
#include "utils.hpp"
#include "vector.hpp"

SV_NAMESPACE_BEGIN

Vector3f SInteraction::Le(const Vector3f &w) const {
  AreaLight *area_light = m_primitive->getAreaLight();
  return area_light ? area_light->L(*this, w) : Vector3f(0.0);
}

Ray Interaction::SpawnRay(const Vector3f &d) const {
  const auto o = OffsetRayOrigin(m_p, m_ns, d);
  return {o, Normalize(d)};
}

Ray Interaction::SpawnRayTo(const Vector3f &p) const {
  Float      norm = (p - m_p).norm();
  auto       d    = (p - m_p) / norm;
  const auto o    = OffsetRayOrigin(m_p, m_ns, d);
  return {o, d, norm - SHADOW_EPS};
}

Ray Interaction::SpawnRayTo(const Interaction &it) const {
  return SpawnRayTo(it.m_p);
}

SV_NAMESPACE_END
