#ifndef __RAY_HPP__
#define __RAY_HPP__

#include <sstream>

#include "fwd.hpp"
#include "vector.hpp"

SV_NAMESPACE_BEGIN

class Volume;
class Ray {
public:
  Ray() : m_tMax(INF) {}
  Ray(const Vector3f &o, const Vector3f &d, Float tMax = INF,
      const Volume *volume = nullptr)
      : m_o(o), m_d(d), m_tMax(tMax), m_volume(volume) {}
  Vector3f    at(Float t) const { return m_o + m_d * t; }
  Vector3f    operator()(Float t) const { return at(t); }
  std::string toString() const {
    std::ostringstream oss;
    oss << "[o=" << m_o << ", d=" << m_d << ", tMax=" << m_tMax << "]";
    return oss.str();
  }

  friend std::ostream &operator<<(std::ostream &os, const Ray &r) {
    os << r.toString();
    return os;
  }

  Vector3f      m_o, m_d;
  mutable Float m_tMax;
  const Volume *m_volume;
  // medium

private:
};

SV_NAMESPACE_END

#endif
