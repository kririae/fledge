#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <memory>

#include "fwd.hpp"
#include "interaction.hpp"
#include "vector.hpp"

SV_NAMESPACE_BEGIN

inline Vector3f SphericalDirection(Float sin_theta, Float cos_theta,
                                   Float phi) {
  return {sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta};
}

// Simple math utils
inline Float CosTheta(const Vector3f &w) {
  return w.z();
}
inline Float Cos2Theta(const Vector3f &w) {
  return w.z() * w.z();
}
inline Float AbsCosTheta(const Vector3f &w) {
  return std::abs(w.z());
}
inline Float Sin2Theta(const Vector3f &w) {
  return std::max((Float)0, (Float)1 - Cos2Theta(w));
}
inline Float SinTheta(const Vector3f &w) {
  return std::sqrt(Sin2Theta(w));
}
inline Float TanTheta(const Vector3f &w) {
  return SinTheta(w) / CosTheta(w);
}
inline Float Tan2Theta(const Vector3f &w) {
  return Sin2Theta(w) / Cos2Theta(w);
}
inline Float CosPhi(const Vector3f &w) {
  Float sin_theta = SinTheta(w);
  return (sin_theta == 0) ? 1 : std::clamp<Float>(w.x() / sin_theta, -1, 1);
}
inline Float SinPhi(const Vector3f &w) {
  Float sin_theta = SinTheta(w);
  return (sin_theta == 0) ? 0 : std::clamp<Float>(w.y() / sin_theta, -1, 1);
}
inline Float Cos2Phi(const Vector3f &w) {
  return CosPhi(w) * CosPhi(w);
}
inline Float Sin2Phi(const Vector3f &w) {
  return SinPhi(w) * SinPhi(w);
}

inline void CoordinateSystem(const Vector3f &v1, Vector3f &v2, Vector3f &v3) {
  if (std::abs(v1.x()) > std::abs(v1.y()))
    v2 = Vector3f(-v1.z(), 0, v1.x()) /
         std::sqrt(v1.x() * v1.x() + v1.z() * v1.z());
  else
    v2 = Vector3f(0, v1.z(), -v1.y()) /
         std::sqrt(v1.y() * v1.y() + v1.z() * v1.z());
  v3 = v1.cross(v2);
}

// copied from previous implementation
inline Vector2f ConcentricSampleDisk(const Vector2f &u) {
  Float ux = u.x() * 2.0 - 1.0, uy = u.y() * 2.0 - 1.0;
  if (ux == 0 && uy == 0) return {0, 0};
  Float theta, r;
  if (abs(ux) > abs(uy)) {
    r     = ux;
    theta = PI_OVER4 * (uy / ux);
  } else {
    r     = uy;
    theta = PI_OVER2 - (PI_OVER4 * (ux / uy));
  }
  return Vector2f(cosf(theta), sinf(theta)) * r;
}

inline Vector3f CosineSampleHemisphere(const Vector2f &u) {
  Vector2f d = ConcentricSampleDisk(u);
  Float    z =
      sqrt(std::max(static_cast<Float>(0), 1 - d[0] * d[0] - d[1] * d[1]));
  return {d[0], d[1], z};
}

inline Vector2f UniformSampleTriangle(const Vector2f &u) {
  Float su0 = sqrt(u.x());
  return {1 - su0, u.y() * su0};
}

inline Vector3f UniformSampleSphere(const Vector2f &u) {
  Float theta     = acos(1 - 2 * u.y());
  Float phi       = 2 * PI * u.x();
  Float sin_theta = sin(theta), cos_theta = cos(theta);
  Float sin_phi = sin(phi), cos_phi = cos(phi);
  return {sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
}

inline Float HGP(const Vector3f &wi, const Vector3f &wo, Float g) {
  Float cos_theta = Dot(wi, wo);
  Float denom     = 1 + g * g + 2 * g * cos_theta;
  return (1 - g * g) / (denom * sqrt(denom)) * INV_4PI;
}

// The following two functions are adopted from rt-render with little
// modification
// (u, v) are the random values
inline Float HGSampleP(Vector3f &wo, Vector3f &wi, Float u, Float v, Float g) {
  Float cos_theta;
  if (abs(g) < 1e-3) {
    cos_theta = 1 - 2 * u;
  } else {
    Float tmp = (1 - g * g) / (1 + g - 2 * g * u);
    cos_theta = -(1 + g * g - tmp * tmp) / (2 * g);
  }

  Float    phi       = 2 * PI * v;
  Float    sin_theta = sqrt(fmax(0, 1 - cos_theta * cos_theta));
  Vector3f v1, v2;
  CoordinateSystem(wo, v1, v2);
  wi = sin_theta * cos(phi) * v1 + sin_theta * sin(phi) * v2 + cos_theta * wo;
  return HGP(wi, wo, g);
}
inline Vector3f Reflect(const Vector3f &wo, const Vector3f &n) {
  return -wo + 2 * Dot(wo, n) * n;
}
inline bool SameHemisphere(const Vector3f &w, const Vector3f &wp) {
  return w.z() * wp.z() > 0;
}

SV_NAMESPACE_END

#endif
