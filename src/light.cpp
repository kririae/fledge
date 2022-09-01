#include "light.hpp"

#include <memory>
#include <vector>

#include "common/aabb.h"
#include "common/math_utils.h"
#include "common/ray.h"
#include "common/vector.h"
#include "debug.hpp"
#include "distribution.hpp"
#include "fledge.h"
#include "interaction.hpp"
#include "scene.hpp"
#include "texture.hpp"

FLG_NAMESPACE_BEGIN

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
  return Vector3f(0.0);
}

Float Light::pdfLe() const {
  return 0.0;
}

Vector3f Light::Le(const Ray &) const {
  return Vector3f(0.0);
}

AreaLight::AreaLight(const std::shared_ptr<Shape> &shape, const Vector3f &Le)
    : m_shape(shape), m_Le(Le) {}

Vector3f AreaLight::sampleLi(const Interaction &ref, const Vector2f &u,
                             Vector3f &wi, Float &pdf,
                             Interaction &sample) const {
  // m_shape->sample (ref, u, pdf) will return the pdf corresponds to
  // d\omega(actually, the p(w) should be put in *pdf*).
  sample = m_shape->sample(ref, u, pdf);
  if (pdf == 0) return Vector3f(0.0);
  wi = (sample.m_p - ref.m_p).normalized();
  return L(sample, -wi);
}

Float AreaLight::pdfLi(const Interaction &isect) const {
  return m_shape->pdf(isect);
}

Vector3f AreaLight::L(const Interaction &isect, const Vector3f &w) const {
  // isect.wo must be initialized
  return isect.m_ng.dot(w) > 0 ? m_Le : Vector3f(0.0);
}

InfiniteAreaLight::InfiniteAreaLight(const Vector3f &color)
    : m_tex(std::make_shared<ConstTexture>(color)) {
  SLog("InfiniteAreaLight is initialized with color=(%f, %f, %f)", color[0],
       color[1], color[2]);
  m_worldCenter = Vector3f(0.0);
  m_worldRadius = 0;
  m_dist        = std::make_shared<Dist2D>(std::vector<Float>(1).data(), 1, 1);
}

InfiniteAreaLight::InfiniteAreaLight(const std::string &filename)
    : InfiniteAreaLight(std::make_shared<ImageTexture>(filename)) {}

InfiniteAreaLight::InfiniteAreaLight(const std::shared_ptr<Texture> &tex)
    : m_tex(tex) {
  SLog("InfiniteAreaLight is initialized with Texture object or filename");
  m_worldCenter = Vector3f(0.0);
  m_worldRadius = 0;

  std::vector<Float> f(NU * NV, 0);
  for (int v = 0; v < NV; ++v)
    for (int u = 0; u < NU; ++u)
      f[v * NU + u] = m_tex->eval(Float(u) / NU, Float(v) / NV).norm();
  m_dist = std::make_shared<Dist2D>(f.data(), NU, NV);
}

void InfiniteAreaLight::preprocess(const Scene &scene) {
  scene.getBound().boundSphere(m_worldCenter, m_worldRadius);
  // TODO: exceed
  m_worldRadius *= 1000;  // scale up the world radius
  SLog("m_worldRadius=%f", m_worldRadius);
  LVec3(m_worldCenter);
}

Vector3f InfiniteAreaLight::sampleLi(const Interaction &ref, const Vector2f &u,
                                     Vector3f &wi, Float &pdf,
                                     Interaction &sample) const {
  Vector2f uv    = m_dist->sampleC(u, pdf);
  Float    phi   = uv[0] * 2 * PI;
  Float    theta = uv[1] * PI;
  auto     dir   = SphericalDirection(std::sin(theta), std::cos(theta), phi);
  pdf /= (2 * PI * PI * std::sin(theta));
#if 1
  dir = UniformSampleSphere(u);
  pdf = 0.5 * INV_2PI;
#endif
  dir = LightToWorld(dir);

  C(pdf, uv, dir);
  auto p     = m_worldCenter + dir * 2 * m_worldRadius;
  sample.m_p = p;

  wi = (sample.m_p - ref.m_p).normalized();
  // return Vector3f(pdf) * 5;
  return Le(Ray{ref.m_p, wi});
}

Float InfiniteAreaLight::pdfLi(const Interaction &it) const {
  TODO();
  return 0.5 * INV_PI;
}

Vector3f InfiniteAreaLight::Le(const Ray &ray) const {
  auto dir = WorldToLight(ray.m_d).normalized();
  C(dir);
  Float phi   = SphericalPhi(dir);
  Float theta = SphericalTheta(dir);
  C(phi, theta);
  return m_tex->eval(phi * INV_2PI, theta * INV_PI);
}

FLG_NAMESPACE_END
