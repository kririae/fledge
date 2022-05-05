#include "light.hpp"

SV_NAMESPACE_BEGIN

// interface for infinite light
Vector3f Light::sampleLe() const {
  return Vector3f::Zero();
}

Float Light::pdfLe() const {
  return 0.0;
}

AreaLight::AreaLight(const std::shared_ptr<Shape> &shape, const Vector3f &Le)
    : m_shape(shape), m_Le(Le) {}

Vector3f AreaLight::sampleLi(const Interaction &ref, const Vector2f &u,
                             Vector3f &wi, Float &pdf) {
  Interaction isect = m_shape->sample(u);
  TODO();
  return Vector3f::Zero();
}

Float AreaLight::pdfLi(const Interaction &isect) {
  return m_shape->pdf(isect);
}

Vector3f AreaLight::L(const Interaction &isect) const {
  // isect.wo must be initialized
  return isect.m_n.dot(isect.m_n) > 0 ? m_Le : Vector3f::Zero();
}

SV_NAMESPACE_END
