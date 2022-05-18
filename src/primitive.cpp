#include "primitive.hpp"

#include <memory>

#include "aabb.hpp"
#include "interaction.hpp"
#include "light.hpp"
#include "ray.hpp"
#include "shape.hpp"

SV_NAMESPACE_BEGIN

ShapePrimitive::ShapePrimitive(const std::shared_ptr<Shape>     &shape,
                               const std::shared_ptr<Material>  &material,
                               const std::shared_ptr<AreaLight> &areaLight)
    : m_shape(shape), m_material(material), m_areaLight(areaLight) {
  // assert(m_areaLight->m_shape.get() == m_shape.get());
}

AABB ShapePrimitive::getBound() const {
  return m_shape->getBound();
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

Material *ShapePrimitive::getMaterial() const {
  return m_material.get();
}

SV_NAMESPACE_END
