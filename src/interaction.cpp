#include "fwd.hpp"
#include "integrator.hpp"
#include "light.hpp"
#include "primitive.hpp"

SV_NAMESPACE_BEGIN

Vector3f SInteraction::Le(const Vector3f &w) const {
  AreaLight *areaLight = m_primitive->getAreaLight();
  return areaLight ? areaLight->L(*this, w) : Vector3f::Zero();
}

SV_NAMESPACE_END
