#include "fwd.hpp"
#include "integrator.hpp"
#include "light.hpp"
#include "primitive.hpp"

SV_NAMESPACE_BEGIN

// cannot be solved with forward declaration
Vector3f SInteraction::Le() const {
  AreaLight *areaLight = m_primitive->getAreaLight();
  return areaLight ? areaLight->L(*this) : Vector3f::Zero();
}

SV_NAMESPACE_END
