#include "material.hpp"

#include "fwd.hpp"
#include "utils.hpp"

SV_NAMESPACE_BEGIN
CoordinateTransition::CoordinateTransition(const Vector3f &normal)
    : m_normal(normal) {
  m_normal.normalize();
  if (abs(m_normal[0]) > abs(m_normal[2])) {
    m_binormal = Vector3f(-m_normal[1], m_normal[0], 0);
  } else {
    m_binormal = Vector3f(0, -m_normal[2], m_normal[1]);
  }
  m_binormal.normalize();
  m_tangent = m_binormal.cross(m_normal);
}

Vector3f CoordinateTransition::WorldToLocal(const Vector3f &p) const {
  return Vector3f{p.dot(m_tangent), p.dot(m_binormal), p.dot(m_normal)};
}

Vector3f CoordinateTransition::LocalToWorld(const Vector3f &p) const {
  return m_tangent * p[0] + m_binormal * p[1] + m_normal * p[2];
}

DiffuseMaterial::DiffuseMaterial(const Vector3f &albedo) : m_albedo(albedo) {}
Vector3f DiffuseMaterial::f(const Vector3f &wo, const Vector3f &wi,
                            const Vector2f &u, const Vector2f &uv,
                            const CoordinateTransition &trans) const {
  return m_albedo * INV_PI;
}

Float DiffuseMaterial::pdf(const Vector3f &wo, const Vector3f &wi,
                           const CoordinateTransition &trans) const {
  Vector3f l_wo = trans.WorldToLocal(wo);
  Vector3f l_wi = trans.WorldToLocal(wi);
  return l_wo[2] * l_wi[2] > 0 ? abs(l_wi[2]) * INV_PI : 0;
}

Vector3f DiffuseMaterial::sampleF(const Vector3f &wo, Vector3f &wi, Float &_pdf,
                                  const Vector2f &u, const Vector2f &uv,
                                  const CoordinateTransition &trans) const {
  wi = CosineSampleHemisphere(u);
  if (wo[2] < 0) wi[2] *= -1;
  _pdf = pdf(wo, wi, trans);
  return f(wo, wi, u, uv, trans);
}

SV_NAMESPACE_END
