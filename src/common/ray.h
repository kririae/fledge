#ifndef __RAY_H__
#define __RAY_H__

#include <sstream>
#include <type_traits>

#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

#if 0
F_CPU_GPU
inline Vector3f OffsetRayOrigin(const Vector3f &p, const Vector3f &n,
                                const Vector3f &dir) {
  // This function is intended to be called when spawning rays.
  // It will assure that the modified ray will not intersect with the geometry
  // near p
  // Out strategy is to move the origin towards the target direction along
  // normal
  Vector3f offset = n;
  if (Dot(n, dir) <= 0) offset = -n;
  return p + offset * SHADOW_EPS;
}
#else
namespace detail_ {
inline float __host_int_as_float(int a) {
  union {
    int   m_a;
    float m_b;
  } u;
  u.m_a = a;
  return u.m_b;
}
inline int __host_float_as_int(float b) {
  union {
    int   m_a;
    float m_b;
  } u;
  u.m_b = b;
  return u.m_a;
}
}  // namespace detail_
/**
 * @brief Offset the ray to avoid self-intersection
 * @note This implementation is from
 * https://research.nvidia.com/publication/2019-03_fast-and-robust-method-avoiding-self-intersection,
 * which provides a way to adaptively construct the new position without
 * tweaking the SHADOW_NORMAL param.
 * The observation is that, using a fixed EPS is not scene invariant and scale
 * invariant. We know that floating-point arithmetics's *relative accuracy*
 * remains almost invariant, but its *absolute accuracy* does not. So a scene's
 * EPS with an characteristic size of 10 is absolutely different from a size of
 * 1e9. That is, the absolute accuracy of intersecting a distant triangle is
 * much more lower. So int arithmetics is used, in which the absolute accuracy
 * is maintained.
 *
 * @param p The ray's original position
 * @param n The *geometry* normal
 * @param dir The ray's direction
 * @return Vector3f representing the ray's original position after offset
 */
F_CPU_GPU inline Vector3f OffsetRayOrigin(const Vector3f &p, const Vector3f &n,
                                          const Vector3f &dir) {
#ifndef __CUDA_CC__
  // Definition of integer arithmetics functions
  const auto &int_as_float = detail_::__host_int_as_float;
  const auto &float_as_int = detail_::__host_float_as_int;
#endif
  if constexpr (std::is_same_v<Float, float>) {
    constexpr float origin      = 1 / 32.0f;
    constexpr float float_scale = 1 / 65536.0f;
    constexpr float int_scale   = 256.0f;
    // Point the offset towards the direction that leaves the surface
    Vector3f offset = n;
    if (Dot(n, dir) <= 0) offset = -n;
    Vector3d offset_int = (offset * int_scale).cast<int, 3>();
    Vector3f p_int{
        int_as_float(float_as_int(p.x()) +
                     ((p.x() < 0) ? -offset_int.x() : offset_int.x())),
        int_as_float(float_as_int(p.y()) +
                     ((p.y() < 0) ? -offset_int.y() : offset_int.y())),
        int_as_float(float_as_int(p.z()) +
                     ((p.z() < 0) ? -offset_int.z() : offset_int.z()))};
    return {
        fabsf(p.x()) < origin ? p.x() + float_scale * offset.x() : p_int.x(),
        fabsf(p.y()) < origin ? p.y() + float_scale * offset.y() : p_int.y(),
        fabsf(p.z()) < origin ? p.z() + float_scale * offset.z() : p_int.z()};
  } else {
    SErr("OffsetRayOrigin for double is not implemented");
  }
}
#endif

class Volume;

struct Ray {
public:
  F_CPU_GPU Ray() : m_tMax(INF), m_volume(nullptr) {}
  F_CPU_GPU Ray(const Vector3f &o, const Vector3f &d, Float tMax = INF,
                Volume const *volume = nullptr)
      : m_o(o), m_d(d), m_tMax(tMax), m_volume(volume) {}
  F_CPU_GPU Ray(const Ray &ray) {
    // explicitly copy all the data
    m_o = ray.m_o, m_d = ray.m_d;
    m_tMax = ray.m_tMax, m_volume = ray.m_volume;
  }
  F_CPU_GPU Vector3f at(Float t) const { return m_o + m_d * t; }
  F_CPU_GPU Vector3f operator()(Float t) const { return at(t); }
  F_CPU_GPU Ray     &operator=(const Ray &rhs) {
        m_o = rhs.m_o, m_d = rhs.m_d;
        m_tMax = rhs.m_tMax, m_volume = rhs.m_volume;
        return *this;
  }
  std::string toString() const {
    std::ostringstream oss;
    oss << "[o=" << m_o << ", d=" << m_d << ", tMax=" << m_tMax << "]";
    return oss.str();
  }
  friend std::ostream &operator<<(std::ostream &os, const Ray &r) {
    os << r.toString();
    return os;
  }

  Vector3f              m_o, m_d;
  mutable Float         m_tMax;
  mutable Volume const *m_volume{nullptr};
  // medium

private:
};

FLG_NAMESPACE_END

#endif
