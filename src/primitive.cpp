#include "primitive.hpp"

SV_NAMESPACE_BEGIN

ShapePrimitive::ShapePrimitive(const std::shared_ptr<Shape>     &shape,
                               const std::shared_ptr<AreaLight> &areaLight)
    : m_shape(shape), m_areaLight(areaLight) {
  assert(m_areaLight->m_shape.get() == m_shape.get());
}

bool ShapePrimitive::intersect(const Ray &ray, SInteraction &isect) const {
  Float tHit;
  if (!m_shape->intersect(ray, tHit, isect)) return false;
  ray.m_tMax        = tHit;
  isect.m_primitive = this;
  return true;
}

// if the areaLight actually exists
AreaLight *ShapePrimitive::getAreaLight() const {
  return m_areaLight.get();
}

SV_NAMESPACE_END
