#include "light.hpp"

#include <memory>

#include "aabb.hpp"
#include "debug.hpp"
#include "fwd.hpp"
#include "interaction.hpp"
#include "ray.hpp"
#include "scene.hpp"
#include "texture.hpp"
#include "utils.hpp"

SV_NAMESPACE_BEGIN

static Float SphericalPhi(const Vector3f &v) {
  Float p = atan2(v[1], v[0]);
  return (p < 0) ? (p + 2 * PI) : p;
}

static Float SphericalTheta(const Vector3f &v) {
  return acos(
      std::clamp(v[2], static_cast<Float>(-1.0), static_cast<Float>(1.0)));
}

static Vector3f LightToWorld(const Vector3f &w) {
  return {-w[1], w[2], -w[0]};
}

static Vector3f WorldToLight(const Vector3f &w) {
  return {-w[2], -w[0], w[1]};
}

void Light::preprocess(const Scene &scene) {
  TODO();
}

// interface for infinite light
Vector3f Light::sampleLe() const {
  return Vector3f::Zero();
}

Float Light::pdfLe() const {
  return 0.0;
}

Vector3f Light::Le(const Ray &) const {
  return Vector3f::Zero();
}

AreaLight::AreaLight(const std::shared_ptr<Shape> &shape, const Vector3f &Le)
    : m_shape(shape), m_Le(Le) {}

Vector3f AreaLight::sampleLi(const Interaction &ref, const Vector2f &u,
                             Vector3f &wi, Float &pdf,
                             Interaction &sample) const {
  // m_shape->sample (ref, u, pdf) will return the pdf corresponds to
  // d\omega(actually, the p(w) should be put in *pdf*).
  sample = m_shape->sample(ref, u, pdf);
  if (pdf == 0) return Vector3f::Zero();
  wi = (sample.m_p - ref.m_p).normalized();
  return L(sample, -wi);
}

Float AreaLight::pdfLi(const Interaction &isect) const {
  return m_shape->pdf(isect);
}

Vector3f AreaLight::L(const Interaction &isect, const Vector3f &w) const {
  // isect.wo must be initialized
  return isect.m_ng.dot(w) > 0 ? m_Le : Vector3f::Zero();
}

InfiniteAreaLight::InfiniteAreaLight(const Vector3f &color)
    : m_tex(std::make_shared<ConstTexture>(color)) {
  SLog("InfiniteAreaLight is initialized with color=(%f, %f, %f)", color[0],
       color[1], color[2]);
  m_worldCenter = Vector3f::Zero();
  m_worldRadius = 0;
}

InfiniteAreaLight::InfiniteAreaLight(const std::string &filename)
    : m_tex(std::make_shared<ImageTexture>(filename)) {
  SLog("InfiniteAreaLight is initialized with filename=(%s)", filename.c_str());
  m_worldCenter = Vector3f::Zero();
  m_worldRadius = 0;
}

InfiniteAreaLight::InfiniteAreaLight(const std::shared_ptr<Texture> &tex)
    : m_tex(tex) {
  SLog("InfiniteAreaLight is initialized with Texture object");
  m_worldCenter = Vector3f::Zero();
  m_worldRadius = 0;
}

void InfiniteAreaLight::preprocess(const Scene &scene) {
  scene.getBound().boundSphere(m_worldCenter, m_worldRadius);
  SLog("m_worldRadius=%f", m_worldRadius);
  LVec3(m_worldCenter);
  m_worldRadius *= 2;
}

Vector3f InfiniteAreaLight::sampleLi(const Interaction &ref, const Vector2f &u,
                                     Vector3f &wi, Float &pdf,
                                     Interaction &sample) const {
  auto dir   = UniformSampleSphere(u);
  auto p     = m_worldCenter + dir * 2 * m_worldRadius;
  sample.m_p = p;
  pdf        = 0.5 * INV_2PI;

  wi = (sample.m_p - ref.m_p).normalized();
  return Le(Ray{ref.m_p, wi});
}

Float InfiniteAreaLight::pdfLi(const Interaction &) const {
  return 0.5 * INV_PI;
}

Vector3f InfiniteAreaLight::Le(const Ray &ray) const {
  auto dir = WorldToLight(ray.m_d).normalized();
  C(dir);
  Float phi   = SphericalPhi(dir);
  Float theta = SphericalTheta(dir);
  return m_tex->eval(C(phi) * INV_2PI, C(theta) * INV_PI);
}

SV_NAMESPACE_END
