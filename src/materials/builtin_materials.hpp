#ifndef _BUILTIN_MATERIALS_HPP_
#define _BUILTIN_MATERIALS_HPP_

#include "common/math_utils.h"
#include "common/vector.h"
#include "fledge.h"
#include "material_base.hpp"

FLG_NAMESPACE_BEGIN

namespace detail_ {
// Exactly the code from PBRT ;)
// https://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models
struct BeckmannDistribution {
  static Float roughnessToAlpha(Float roughness) {
    roughness = std::max(roughness, (Float)1e-3);
    Float x   = std::log(roughness);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
           0.000640711f * x * x * x * x;
  }

  BeckmannDistribution(Float alpha) : m_alpha_x(alpha), m_alpha_y(alpha) {}
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
}  // namespace detail_

#define MAKE_MATERIAL_IMPL                                            \
  F_CPU_GPU Vector3f f_impl(const Vector3f &wo, const Vector3f &wi,   \
                            const Vector2f &uv) const override;       \
  F_CPU_GPU Float    pdf_impl(const Vector3f &wo, const Vector3f &wi) \
      const override;                                                 \
  F_CPU_GPU Vector3f sampleF_impl(const Vector3f &wo, Vector3f &wi,   \
                                  Float &pdf, const Vector2f &u,      \
                                  const Vector2f &uv) const override; \
  F_CPU_GPU bool     isDelta_impl() const override;                   \
  F_CPU_GPU Vector3f getAlbedo_impl() const override;

class DiffuseMaterial : public MaterialDispatcher {
public:
  MAKE_MATERIAL_IMPL
  F_CPU_GPU DiffuseMaterial(const Vector3f &albedo);

private:
  Vector3f m_albedo;
};

class MicrofacetMaterial : public MaterialDispatcher {
public:
  // MAKE_MATERIAL_IMPL
  // roughness, k: absorption coefficient
  F_CPU_GPU MicrofacetMaterial(const Vector3f &R, Float roughness = 0.4,
                               const Vector3f &k = Vector3f(2.0));

private:
  Vector3f                      m_R, m_k;
  Float                         m_roughness;
  detail_::BeckmannDistribution m_dist;
};

class TransmissionMaterial : public MaterialDispatcher {
public:
  // MAKE_MATERIAL_IMPL
  F_CPU_GPU TransmissionMaterial(Float etaI, Float etaT);

private:
  Float m_etaI, m_etaT;
};

template <typename T, typename... Args>
inline MaterialDispatcher *MakeMaterialInstance(Resource &resource,
                                                Args... args) {
  T *material = resource.alloc<T>(std::forward<Args>(args)...);
  return resource.alloc<MaterialDispatcher>(material);
}

FLG_NAMESPACE_END

#endif
