#include "fwd.hpp"
#include "integrator.hpp"
#include "light.hpp"
#include "primitive.hpp"
#include "utils.hpp"
#include "vector.hpp"

FLG_NAMESPACE_BEGIN

Vector3f SInteraction::Le(const Vector3f &w) const {
  AreaLight *area_light = m_primitive->getAreaLight();
  return area_light ? area_light->L(*this, w) : Vector3f(0.0);
}

FLG_NAMESPACE_END
