#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include "fwd.hpp"

SV_NAMESPACE_BEGIN

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
  Float theta    = acos(1 - 2 * u.y());
  Float phi      = 2 * PI * u.x();
  Float sinTheta = sin(theta), cosTheta = cos(theta);
  Float sinPhi = sin(phi), cosPhi = cos(phi);
  return {sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
}

SV_NAMESPACE_END

#endif
