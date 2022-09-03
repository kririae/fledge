#ifndef __FRESNEL_HPP__
#define __FRESNEL_HPP__

#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

/* For the following three functions,
 * **Assume** that etaI is in the side of n, etaT is in the side of -n,
 * and cosTheta do not need to be adjusted */
// (Float, Float, Float) -> Float
F_CPU_GPU inline Float FresnelDielectric(Float cosThetaI, Float etaI,
                                         Float etaT) {
  cosThetaI     = abs(cosThetaI);
  bool entering = cosThetaI > 0.f;
  if (!entering) {
    auto tmp = etaI;
    etaI     = etaT;
    etaT     = tmp;
  }

  Float sin_theta_i =
      std::sqrt(std::max(static_cast<Float>(0.0), 1 - cosThetaI * cosThetaI));
  Float sin_theta_t = etaI / etaT * sin_theta_i;  // snell's law
  if (sin_theta_t >= 1) {
    return 1.;
  }  // Total reflection

  Float cos_theta_t = std::sqrt(
      std::max(static_cast<Float>(0.0), 1 - sin_theta_t * sin_theta_t));

  // The formula and code exactly from PBRT
  Float rparl = ((etaT * cosThetaI) - (etaI * cos_theta_t)) /
                ((etaT * cosThetaI) + (etaI * cos_theta_t));
  Float rperp = ((etaI * cosThetaI) - (etaT * cos_theta_t)) /
                ((etaI * cosThetaI) + (etaT * cos_theta_t));
  return (rparl * rparl + rperp * rperp) / 2;
}

F_CPU_GPU inline Float FresnelSchlick(Float cos_theta_i, Float etaI,
                                      Float etaT) {
  // Following the assumption to do some simplification
  cos_theta_i = std::abs(cos_theta_i);
  cos_theta_i = std::max<Float>(0.0, cos_theta_i);
  cos_theta_i = std::min<Float>(1.0, cos_theta_i);
  Float R0    = (etaI - etaT) / (etaI + etaT);
  R0          = R0 * R0;
  return R0 + (1 - R0) * pow(1 - cos_theta_i, 5);
}

F_CPU_GPU inline Vector3f FresnelConductor(Float           cos_theta_i,
                                           const Vector3f &etaI,
                                           const Vector3f &etaT,
                                           const Vector3f &k) {
  cos_theta_i    = std::clamp<Float>(cos_theta_i, -1, 1);
  Vector3f eta   = etaT / etaI;
  Vector3f eta_k = k / etaI;

  Float    cos_theta_i2 = cos_theta_i * cos_theta_i;
  Float    sin_theta_i2 = 1 - cos_theta_i2;
  Vector3f eta2         = eta * eta;
  Vector3f eta_k2       = eta_k * eta_k;

  Vector3f t0         = eta2 - eta_k2 - Vector3f(sin_theta_i2);
  Vector3f a2_plus_b2 = (t0 * t0 + 4 * eta2 * eta_k2);
  a2_plus_b2[0]       = std::sqrt(a2_plus_b2[0]);
  a2_plus_b2[1]       = std::sqrt(a2_plus_b2[1]);
  a2_plus_b2[2]       = std::sqrt(a2_plus_b2[2]);
  Vector3f t1         = a2_plus_b2 + Vector3f(cos_theta_i2);
  Vector3f a          = 0.5f * (a2_plus_b2 + t0);
  a[0]                = std::sqrt(a[0]);
  a[1]                = std::sqrt(a[1]);
  a[2]                = std::sqrt(a[2]);
  Vector3f t2         = (Float)2 * cos_theta_i * a;
  Vector3f rs         = (t1 - t2) / (t1 + t2);

  Vector3f t3 =
      cos_theta_i2 * a2_plus_b2 + Vector3f(sin_theta_i2 * sin_theta_i2);
  Vector3f t4 = t2 * sin_theta_i2;
  Vector3f rp = rs * (t3 - t4) / (t3 + t4);

  return 0.5 * (rp + rs);
}

FLG_NAMESPACE_END
#endif
