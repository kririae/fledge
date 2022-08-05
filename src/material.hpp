#ifndef __MATERIAL_HPP__
#define __MATERIAL_HPP__

#include "fwd.hpp"
#include "utils.hpp"

SV_NAMESPACE_BEGIN

class CoordinateTransition {
public:
  CoordinateTransition(const Vector3f &normal);
  Vector3f WorldToLocal(const Vector3f &p) const;
  Vector3f LocalToWorld(const Vector3f &p) const;

private:
  Vector3f m_normal, m_tangent, m_binormal;
};

enum class EMaterialType : int { DIFFUSE = 0, EMPTY };
class Material {
public:
  virtual ~Material() = default;

  virtual Vector3f f(const Vector3f &wo, const Vector3f &wi, const Vector2f &uv,
                     const CoordinateTransition &trans) const       = 0;
  virtual Float    pdf(const Vector3f &wo, const Vector3f &wi,
                       const CoordinateTransition &trans) const     = 0;
  virtual Vector3f sampleF(const Vector3f &wo, Vector3f &wi, Float &pdf,
                           const Vector2f &u, const Vector2f &uv,
                           const CoordinateTransition &trans) const = 0;
  EMaterialType    m_matType;
};

class DiffuseMaterial : public Material {
public:
  DiffuseMaterial(const Vector3f &albedo);
  Vector3f f(const Vector3f &wo, const Vector3f &wi, const Vector2f &uv,
             const CoordinateTransition &trans) const override;
  Float    pdf(const Vector3f &wo, const Vector3f &wi,
               const CoordinateTransition &trans) const override;
  Vector3f sampleF(const Vector3f &wo, Vector3f &wi, Float &pdf,
                   const Vector2f &u, const Vector2f &uv,
                   const CoordinateTransition &trans) const override;

private:
  Vector3f m_albedo;
};

// Exactly the code from PBRT ;)
// https://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models
struct BeckmannDistribution {
  static Float roughnessToAlpha(Float roughness) {
    roughness = std::max(roughness, (Float)1e-3);
    Float x   = std::log(roughness);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
           0.000640711f * x * x * x * x;
  }

  BeckmannDistribution(Float alpha_x, Float alpha_y)
      : m_alpha_x(alpha_x), m_alpha_y(alpha_y) {}
  Float D(const Vector3f &wh) const {
    Float tan2_theta = Tan2Theta(wh);
    if (std::isinf(tan2_theta)) return 0.;
    Float cos4_theta = Cos2Theta(wh) * Cos2Theta(wh);
    return std::exp(-tan2_theta * (Cos2Phi(wh) / (m_alpha_x * m_alpha_x) +
                                   Sin2Phi(wh) / (m_alpha_y * m_alpha_y))) /
           (PI * m_alpha_x * m_alpha_y * cos4_theta);
  }
  Float lambda(const Vector3f &w) const {
    Float abs_tan_theta = std::abs(TanTheta(w));
    if (std::isinf(abs_tan_theta)) return 0.;
    Float alpha = std::sqrt(Cos2Phi(w) * m_alpha_x * m_alpha_x +
                            Sin2Phi(w) * m_alpha_y * m_alpha_y);
    Float a     = 1 / (alpha * abs_tan_theta);
    if (a >= 1.6f) return 0;
    return (1 - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
  }
  Float G1(const Vector3f &w) const { return 1 / (1 + lambda(w)); }
  Float G(const Vector3f &wo, const Vector3f &wi) const {
    return 1 / (1 + lambda(wo) + lambda(wi));
  }
  Float pdf([[maybe_unused]] const Vector3f &wo, const Vector3f &wh) const {
    return D(wh) * AbsCosTheta(wh);
  }
  Vector3f sampleWh(const Vector3f &wo, const Vector2f &u) const {
    Float tan2_theta, phi;
    if (m_alpha_x == m_alpha_y) {
      Float log_sample = std::log(1 - u[0]);
      C(log_sample);
      tan2_theta = -m_alpha_x * m_alpha_x * log_sample;
      phi        = u[1] * 2 * PI;
    } else {
      // distribution
      Float log_sample = std::log(1 - u[0]);
      C(log_sample);
      phi = std::atan(m_alpha_y / m_alpha_x *
                      std::tan(2 * PI * u[1] + 0.5f * PI));
      if (u[1] > 0.5f) phi += PI;
      Float sin_phi = std::sin(phi), cos_phi = std::cos(phi);
      Float m_alpha_x2 = m_alpha_x * m_alpha_x,
            m_alpha_y2 = m_alpha_y * m_alpha_y;
      tan2_theta       = -log_sample / (cos_phi * cos_phi / m_alpha_x2 +
                                  sin_phi * sin_phi / m_alpha_y2);
    }

    Float cos_theta = 1 / std::sqrt(1 + tan2_theta);
    Float sin_theta = std::sqrt(std::max((Float)0, 1 - cos_theta * cos_theta));
    Vector3f wh     = SphericalDirection(sin_theta, cos_theta, phi);
    if (wo.z() * wh.z() <= 0) wh = -wh;
    return wh;
  }

  Float m_alpha_x, m_alpha_y;
};

class MicrofacetMaterial : public Material {
public:
  MicrofacetMaterial(const Vector3f &R);
  Vector3f f(const Vector3f &wo, const Vector3f &wi, const Vector2f &uv,
             const CoordinateTransition &trans) const override;
  Float    pdf(const Vector3f &wo, const Vector3f &wi,
               const CoordinateTransition &trans) const override;
  Vector3f sampleF(const Vector3f &wo, Vector3f &wi, Float &pdf,
                   const Vector2f &u, const Vector2f &uv,
                   const CoordinateTransition &trans) const override;

private:
  Vector3f m_R;
};

SV_NAMESPACE_END

#endif
