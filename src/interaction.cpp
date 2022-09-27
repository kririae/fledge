#include "common/math_utils.h"
#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"
#include "integrator.hpp"
#include "light.hpp"
#include "primitive.hpp"

FLG_NAMESPACE_BEGIN

Vector3f SInteraction::Le(const Vector3f &w) const {
  AreaLight *area_light = m_primitive->getAreaLight();
  return area_light ? area_light->L(*this, w) : Vector3f(0.0);
}

Ray SInteraction::SpawnRay(const Vector3f &d) const {
  auto res     = Interaction::SpawnRay(d);
  res.m_volume = m_ray.m_volume;  // copy the previous volume
  if (Entering(d, m_ng)) {
    if (m_primitive != nullptr) res.m_volume = m_primitive->getVolume();
  } else {
    res.m_volume = nullptr;
  }

  return res;
}

Ray SInteraction::SpawnRayTo(const Vector3f &p) const {
  auto res     = Interaction::SpawnRayTo(p);
  res.m_volume = m_ray.m_volume;  // copy the previous volume
  if (Entering(res.m_d, m_ng)) {
    if (m_primitive != nullptr) res.m_volume = m_primitive->getVolume();
  } else {
    res.m_volume = nullptr;
  }

  return res;
}

Ray SInteraction::SpawnRayTo(const Interaction &it) const {
  return SInteraction::SpawnRayTo(it.m_p);
}

Ray VInteraction::SpawnRay(const Vector3f &d) const {
  auto res = Interaction::SpawnRay(d);
  assert(Norm(m_ray.m_d) != 0);
  res.m_volume = m_ray.m_volume;  // copy the previous volume
  return res;
}

Ray VInteraction::SpawnRayTo(const Vector3f &p) const {
  auto res = Interaction::SpawnRayTo(p);
  assert(Norm(m_ray.m_d) != 0);
  res.m_volume = m_ray.m_volume;  // copy the previous volume
  return res;
}

Ray VInteraction::SpawnRayTo(const Interaction &it) const {
  return VInteraction::SpawnRayTo(it.m_p);
}

FLG_NAMESPACE_END
