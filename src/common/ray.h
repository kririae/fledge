#ifndef __RAY_H__
#define __RAY_H__

#include <sstream>

#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

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
  mutable Volume const *m_volume;
  // medium

private:
};

FLG_NAMESPACE_END

#endif
