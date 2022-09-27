#include "builtin_materials.hpp"

#include "common/math_utils.h"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

/**
 * This section contains the constructors of builtin materials
 */
F_CPU_GPU
DiffuseMaterial::DiffuseMaterial(const Vector3f &albedo) : m_albedo(albedo) {}

// Yet adapted from PBRT src/core/reflection.cpp
F_CPU_GPU
MicrofacetMaterial::MicrofacetMaterial(const Vector3f &R, Float roughness,
                                       const Vector3f &k)
    : m_R(R),
      m_k(k),
      m_roughness(roughness),
      m_dist(detail_::BeckmannDistribution(
          detail_::BeckmannDistribution::roughnessToAlpha(roughness))) {}

F_CPU_GPU
TransmissionMaterial::TransmissionMaterial(Float etaI, Float etaT)
    : m_etaI(etaI), m_etaT(etaT) {}

/**
 * This section contains the implementations of builtin materials
 */
F_CPU_GPU
Vector3f DiffuseMaterial::f_impl(const Vector3f &wo, const Vector3f &wi,
                                 const Vector2f &uv) const {
  return m_albedo * INV_PI;
}

F_CPU_GPU
Float DiffuseMaterial::pdf_impl(const Vector3f &wo, const Vector3f &wi) const {
  return wo[2] * wi[2] > 0 ? abs(wi[2]) * INV_PI : 0;
}

F_CPU_GPU
Vector3f DiffuseMaterial::sampleF_impl(const Vector3f &wo, Vector3f &wi,
                                       Float &pdf_, const Vector2f &u,
                                       const Vector2f &uv) const {
  wi   = CosineSampleHemisphere(u);
  pdf_ = pdf_impl(wo, wi);
  return f_impl(wo, wi, uv);
}

F_CPU_GPU
bool DiffuseMaterial::isDelta_impl() const {
  return false;
}

F_CPU_GPU
Vector3f DiffuseMaterial::getAlbedo_impl() const {
  return m_albedo;
}

#if 0
Vector3f MicrofacetMaterial::f(const Vector3f &w_wo, const Vector3f &w_wi,
                               const Vector2f             &uv,
                               const CoordinateTransition &trans) const {
  Vector3f wo = trans.WorldToLocal(w_wo);
  Vector3f wi = trans.WorldToLocal(w_wi);
  C(wo, wi);  // validity check

  Float    cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
  Vector3f wh = wi + wo;
  if (cosThetaI == 0 || cosThetaO == 0) return Vector3f(0.0);
  if (wh.x() == 0 && wh.y() == 0 && wh.z() == 0) return Vector3f(0.0);

  wh.normalize();
  Vector3f F = FresnelConductor(wi.dot(wh), Vector3f(1.0), Vector3f(1.0), m_k);
  return (m_R * m_dist.D(wh) * m_dist.G(wo, wi)) * (F) /
         (4 * cosThetaI * cosThetaO);
}

Float MicrofacetMaterial::pdf(const Vector3f &w_wo, const Vector3f &w_wi,
                              const CoordinateTransition &trans) const {
  Vector3f wo = trans.WorldToLocal(w_wo);
  Vector3f wi = trans.WorldToLocal(w_wi);
  Vector3f wh = (wo + wi).stableNormalized();
  return m_dist.pdf(wo, wh) / (4 * wo.dot(wh));
}

Vector3f MicrofacetMaterial::sampleF(const Vector3f &w_wo, Vector3f &w_wi,
                                     Float &pdf, const Vector2f &u,
                                     const Vector2f             &uv,
                                     const CoordinateTransition &trans) const {
  // As usual, begin with coordinate transition
  Vector3f wo = trans.WorldToLocal(w_wo);

  // Sample microfacet orientation $\wh$ and reflected direction $\wi$
  if (wo.z() == 0) return Vector3f(0.0);
  Vector3f wh = m_dist.sampleWh(wo, u);
  if (wo.dot(wh) < 0) return Vector3f(0.0);  // Should be rare
  auto wi = Reflect(wo, wh);
  w_wi    = trans.LocalToWorld(wi);
  if (!SameHemisphere(wo, wi)) return Vector3f(0.0);

  // Compute PDF of _wi_ for microfacet reflection
  pdf = m_dist.pdf(wo, wh) / (4 * wo.dot(wh));
  return f(w_wo, w_wi, Vector2f(0.0), trans);
}

// Transmission Material
Vector3f TransmissionMaterial::f(const Vector3f &w_wo, const Vector3f &w_wi,
                                 const Vector2f             &uv,
                                 const CoordinateTransition &trans) const {
  return Vector3f(0.0);
}

Float TransmissionMaterial::pdf(const Vector3f &w_wo, const Vector3f &w_wi,
                                const CoordinateTransition &trans) const {
  return 0.0;
}

Vector3f TransmissionMaterial::sampleF(
    const Vector3f &w_wo, Vector3f &w_wi, Float &pdf, const Vector2f &u,
    const Vector2f &uv, const CoordinateTransition &trans) const {
  Vector3f wo = trans.WorldToLocal(w_wo), wi;

  bool  entering = CosTheta(wo) > 0;
  Float etaI     = entering ? m_etaI : m_etaT;
  Float etaT     = entering ? m_etaT : m_etaI;

  Float F = FresnelDielectric(CosTheta(wo), etaI, etaT);
  if (u[0] < F) {
    wi   = {-wo.x(), -wo.y(), wo.z()};
    w_wi = trans.LocalToWorld(wi);
    pdf  = F;
    return F / AbsCosTheta(wi);
  } else {
    Vector3f n = CosTheta(wo) > 0 ? Vector3f(0, 0, 1) : Vector3f(0, 0, -1);
    if (!Refract(wo, n, etaI / etaT, wi)) return 0;
    w_wi        = trans.LocalToWorld(wi);
    pdf         = 1 - F;
    Vector3f ft = Vector3f(1 - F);
    ft *= (etaI * etaI) / (etaT * etaT);
    return ft / AbsCosTheta(wi);
  }
}
#endif

FLG_NAMESPACE_END