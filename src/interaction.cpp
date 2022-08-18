#include "fwd.hpp"
#include "integrator.hpp"
#include "light.hpp"
#include "primitive.hpp"
#include "vector.hpp"

SV_NAMESPACE_BEGIN

Vector3f SInteraction::Le(const Vector3f &w) const {
  AreaLight *area_light = m_primitive->getAreaLight();
  return area_light ? area_light->L(*this, w) : Vector3f::Zero();
}

SV_NAMESPACE_END
