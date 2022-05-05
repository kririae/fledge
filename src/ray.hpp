#ifndef __RAY_HPP__
#define __RAY_HPP__

#include "fwd.hpp"

SV_NAMESPACE_BEGIN

class Ray {
public:
  Ray() : m_tMax(INF) {}
  Ray(const Vector3f &o, const Vector3f &d, Float tMax = INF)
      : m_o(o), m_d(d), m_tMax(tMax) {}
  Vector3f             at(Float t) const { return m_o + m_d * t; }
  Vector3f             operator()(Float t) const { return at(t); }
  friend std::ostream &operator<<(std::ostream &os, const Ray &r) {
    os << "[o=" << r.m_o << ", d=" << r.m_d << ", tMax=" << r.m_tMax << "]";
    return os;
  }

  Vector3f      m_o, m_d;
  mutable Float m_tMax;
  // medium

private:
};

SV_NAMESPACE_END

#endif
