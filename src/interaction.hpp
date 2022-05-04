#ifndef __INTERACTION_HPP__
#define __INTERACTION_HPP__

#include "fwd.hpp"
#include "ray.hpp"

SV_NAMESPACE_BEGIN

class Interaction {
public:
  Interaction() = default;
  Interaction(const Vector3f &p, const Vector3f &n, const Vector3f &wo)
      : m_p(p), m_n(n), m_wo(wo) {}
  virtual ~Interaction() = default;
  virtual void reset() {
    m_p = m_n = m_wo = Vector3f::Zero();
    m_t              = INF;
  }
  virtual Ray SpawnRay(const Vector3f &d) { return {m_p, d}; }
  virtual Ray SpawnRayTo(const Vector3f &p) {
    return {m_p, (p - m_p).normalized()};
  }
  virtual Ray SpawnRayTo(const Interaction &it) {
    return {m_p, (it.m_p - m_p).normalized()};
  }

  Vector3f m_p, m_n, m_wo;
  Float    m_t{INF};

private:
};

class SInteraction : public Interaction {
public:
  SInteraction() = default;
  SInteraction(const Vector3f &p, const Vector3f &n, const Vector3f &wo)
      : Interaction(p, n, wo) {}
  virtual ~SInteraction() = default;
};

class VInteraction : public Interaction {
public:
  VInteraction() = default;
  VInteraction(const Vector3f &p, const Vector3f &n, const Vector3f &wo)
      : Interaction(p, n, wo) {}
  virtual ~VInteraction() = default;
};

SV_NAMESPACE_END

#endif
