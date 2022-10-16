#ifndef __INTERSECTOR_HPP__
#define __INTERSECTOR_HPP__

#include "common/aabb.h"
#include "common/math_utils.h"
#include "common/vector.h"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

namespace experimental {
namespace detail_ {
__always_inline bool TriangleIntersect(const Vector3f &a, const Vector3f &b,
                                       const Vector3f &c, const Vector3f &ray_o,
                                       const Vector3f &ray_d, float *thit,
                                       Vector3f *ng) {
  // Naive implementation
  Vector3f e1      = b - a;
  Vector3f e2      = c - a;
  Vector3f p       = Cross(ray_d, e2);
  Float    d       = e1.dot(p);
  Float    inv_det = 1 / d;
  if (d == 0) return false;

  Vector3f t = ray_o - a;
  Float    u = t.dot(p) * inv_det;
  if (u < 0 || u > 1) return false;

  Vector3f q = t.cross(e1);
  Float    v = ray_d.dot(q) * inv_det;
  if (v < 0 || u + v > 1) return false;

  Float t_near = e2.dot(q) * inv_det;
  if (t_near <= 0) return false;

  if (thit != nullptr) *thit = t_near;
  if (ng != nullptr) *ng = Normalize(Cross(e1, e2));
  return true;
}

__always_inline float msub(const float a, const float b, const float c) {
  return a * b - c;
}
/* code from embree3 */
__always_inline Vector3f StableTriangleNormal(const Vector3f &a,
                                              const Vector3f &b,
                                              const Vector3f &c) {
  const float ab_x = a.z() * b.y(), ab_y = a.x() * b.z(), ab_z = a.y() * b.x();
  const float bc_x = b.z() * c.y(), bc_y = b.x() * c.z(), bc_z = b.y() * c.x();
  const Vector3f cross_ab(msub(a.y(), b.z(), ab_x), msub(a.z(), b.x(), ab_y),
                          msub(a.x(), b.y(), ab_z));
  const Vector3f cross_bc(msub(b.y(), c.z(), bc_x), msub(b.z(), c.x(), bc_y),
                          msub(b.x(), c.y(), bc_z));
  const auto     sx = abs(ab_x) < abs(bc_x);
  const auto     sy = abs(ab_y) < abs(bc_y);
  const auto     sz = abs(ab_z) < abs(bc_z);
  return {Select(sx, cross_ab.x(), cross_bc.x()),
          Select(sy, cross_ab.y(), cross_bc.y()),
          Select(sz, cross_ab.z(), cross_bc.z())};
}

/* code from embree3 */
__always_inline bool PlueckerTriangleIntersect(
    const Vector3f &a, const Vector3f &b, const Vector3f &c,
    const Vector3f &ray_o, const Vector3f &ray_d, float *thit, Vector3f *ng) {
  constexpr float ulp = std::numeric_limits<float>::epsilon();
  // From Intel's implementation
  const Vector3f O  = ray_o;
  const Vector3f D  = ray_d;
  const Vector3f v0 = a - O;
  const Vector3f v1 = b - O;
  const Vector3f v2 = c - O;
  const Vector3f e0 = v2 - v0;
  const Vector3f e1 = v0 - v1;
  const Vector3f e2 = v1 - v2;

  const float U   = Dot(Cross(e0, v2 + v0), D);
  const float V   = Dot(Cross(e1, v0 + v1), D);
  const float W   = Dot(Cross(e2, v1 + v2), D);
  const float UVW = U + V + W;
  const float eps = float(ulp) * abs(UVW);
  bool        valid =
      std::min(U, std::min(V, W)) >= -eps || std::max(U, std::max(V, W)) <= eps;
  if (!valid) return false;

  // TODO: use this algorithm for now
  const Vector3f Ng  = StableNormalize(StableTriangleNormal(e0, e1, e2));
  const float    den = Twice(Dot(Ng, D));

  const float T = Twice(Dot(v0, Ng));
  const float t = T / den;  // rcp(den) * T
  valid &= (0 <= t);
  if (!valid) return false;

  valid &= (den != 0);
  if (!valid) return false;

  if (ng != nullptr) *ng = Ng;
  if (thit != nullptr) *thit = t;
  return true;
}

__always_inline bool BoundIntersect(const Vector3f &lower,
                                    const Vector3f &upper,
                                    const Vector3f &ray_o,
                                    const Vector3f &ray_d, float &tnear,
                                    float &tfar) {
  Float t0 = 0, t1 = std::numeric_limits<float>::max();
  for (int i = 0; i < 3; ++i) {
    Float inv_ray_dir = 1 / ray_d[i];
    Float t_near      = (lower[i] - ray_o[i]) * inv_ray_dir;
    Float t_far       = (upper[i] - ray_o[i]) * inv_ray_dir;
    if (t_near > t_far) std::swap(t_near, t_far);
    t0 = t_near > t0 ? t_near : t0;
    t1 = t_far < t1 ? t_far : t1;
    if (t0 > t1) return false;
  }  // for
  tnear = t0, tfar = t1;
  return true;
}
}  // namespace detail_
}  // namespace experimental

FLG_NAMESPACE_END

#endif
