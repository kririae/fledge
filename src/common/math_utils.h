#ifndef __MATH_UTILS_H__
#define __MATH_UTILS_H__

#include <sys/cdefs.h>

#include <memory>
#include <type_traits>

#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

F_CPU_GPU inline Vector3f SphericalDirection(Float sin_theta, Float cos_theta,
                                             Float phi) {
  return {sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta};
}

// Simple math utils
F_CPU_GPU inline Float CosTheta(const Vector3f &w) {
  return w.z();
}
F_CPU_GPU inline Float Cos2Theta(const Vector3f &w) {
  return w.z() * w.z();
}
F_CPU_GPU inline Float AbsCosTheta(const Vector3f &w) {
  return std::abs(w.z());
}
F_CPU_GPU inline Float Sin2Theta(const Vector3f &w) {
  return std::max((Float)0, (Float)1 - Cos2Theta(w));
}
F_CPU_GPU inline Float SinTheta(const Vector3f &w) {
  return std::sqrt(Sin2Theta(w));
}
F_CPU_GPU inline Float TanTheta(const Vector3f &w) {
  return SinTheta(w) / CosTheta(w);
}
F_CPU_GPU inline Float Tan2Theta(const Vector3f &w) {
  return Sin2Theta(w) / Cos2Theta(w);
}
F_CPU_GPU inline Float CosPhi(const Vector3f &w) {
  Float sin_theta = SinTheta(w);
  return (sin_theta == 0) ? 1 : std::clamp<Float>(w.x() / sin_theta, -1, 1);
}
F_CPU_GPU inline Float SinPhi(const Vector3f &w) {
  Float sin_theta = SinTheta(w);
  return (sin_theta == 0) ? 0 : std::clamp<Float>(w.y() / sin_theta, -1, 1);
}
F_CPU_GPU inline Float Cos2Phi(const Vector3f &w) {
  return CosPhi(w) * CosPhi(w);
}
F_CPU_GPU inline Float Sin2Phi(const Vector3f &w) {
  return SinPhi(w) * SinPhi(w);
}

F_CPU_GPU inline void CoordinateSystem(const Vector3f &v1, Vector3f &v2,
                                       Vector3f &v3) {
  if (std::abs(v1.x()) > std::abs(v1.y()))
    v2 = Vector3f(-v1.z(), 0, v1.x()) /
         std::sqrt(v1.x() * v1.x() + v1.z() * v1.z());
  else
    v2 = Vector3f(0, v1.z(), -v1.y()) /
         std::sqrt(v1.y() * v1.y() + v1.z() * v1.z());
  v3 = v1.cross(v2);
}

// copied from previous implementation
F_CPU_GPU inline Vector2f ConcentricSampleDisk(const Vector2f &u) {
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

F_CPU_GPU inline Vector3f CosineSampleHemisphere(const Vector2f &u) {
  Vector2f d = ConcentricSampleDisk(u);
  Float    z =
      sqrt(std::max(static_cast<Float>(0), 1 - d[0] * d[0] - d[1] * d[1]));
  return {d[0], d[1], z};
}

F_CPU_GPU inline Vector2f UniformSampleTriangle(const Vector2f &u) {
  Float su0 = sqrt(u.x());
  return {1 - su0, u.y() * su0};
}

F_CPU_GPU inline Vector3f UniformSampleSphere(const Vector2f &u) {
  Float theta     = acos(1 - 2 * u.y());
  Float phi       = 2 * PI * u.x();
  Float sin_theta = sin(theta), cos_theta = cos(theta);
  Float sin_phi = sin(phi), cos_phi = cos(phi);
  return {sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
}

F_CPU_GPU inline Float HGP(const Vector3f &wi, const Vector3f &wo, Float g) {
  Float cos_theta = Dot(wi, wo);
  Float denom     = 1 + g * g + 2 * g * cos_theta;
  return (1 - g * g) / (denom * sqrt(denom)) * INV_4PI;
}

// The following two functions are adopted from rt-render with little
// modification
// (u, v) are the random values
F_CPU_GPU inline Float HGSampleP(Vector3f &wo, Vector3f &wi, Float u, Float v,
                                 Float g) {
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
F_CPU_GPU inline Vector3f Reflect(const Vector3f &wo, const Vector3f &n) {
  return -wo + 2 * Dot(wo, n) * n;
}
F_CPU_GPU inline bool SameHemisphere(const Vector3f &w, const Vector3f &wp) {
  return w.z() * wp.z() > 0;
}

// Correct implementation from PBRT again
F_CPU_GPU inline bool Refract(const Vector3f &wi, const Vector3f &n, Float eta,
                              Vector3f &wt) {
  Float cos_theta_i  = Dot(n, wi);
  Float sin2_theta_i = std::max(0.f, 1.f - cos_theta_i * cos_theta_i);
  Float sin2_theta_t = eta * eta * sin2_theta_i;
  if (sin2_theta_t >= 1) return false;
  Float cos_theta_t = std::sqrt(1 - sin2_theta_t);
  wt = eta * -wi + (eta * cos_theta_i - cos_theta_t) * Vector3f(n);
  return true;
}

// wi is the leaving ray, wi -> primitive -> wo
F_CPU_GPU inline bool Entering(const Vector3f &wi, const Vector3f &n) {
  return wi.dot(n) < 0;
}

// https://pbr-book.org/3ed-2018/Utilities/Main_Include_File#Mod
template <typename T>
F_CPU_GPU inline T Mod(T a, T b) {
  if constexpr (std::is_same_v<T, Float>) {
    return std::fmod(a, b);
  } else {
    T result = a - (a / b) * b;
    return (T)((result < 0) ? result + b : result);
  }
}

template <typename T>
F_CPU_GPU __always_inline T Twice(const T &a) {
  return a + a;
}

template <typename T>
F_CPU_GPU __always_inline T Select(bool s, const T &a, const T &b) {
  return s ? a : b;
}

F_CPU_GPU __always_inline int Sign(int a) {
  return a == 0 ? 0 : (a > 0 ? 1 : -1);
}

FLG_NAMESPACE_END

#endif
