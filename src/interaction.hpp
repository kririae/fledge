#ifndef __INTERACTION_HPP__
#define __INTERACTION_HPP__

#include "common/math_utils.h"
#include "common/ray.h"
#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

class Primitive;

/**
        Interaction is responsible for handling volume interaction, since it
   holds sufficient information to sling volume onto the spawned ray
 */
class Interaction {
public:
  Interaction() = default;
  Interaction(const Vector3f &p, const Vector3f &ns, const Vector3f &ng,
              const Vector3f &wo)
      : m_p(p), m_ns(ns), m_ng(ng), m_wo(wo) {}
  virtual ~Interaction() = default;
  virtual void reset() {
    m_p = m_ns = m_ng = m_wo = Vector3f(0.0);
    m_ray                    = Ray();
  }
  virtual Ray SpawnRay(const Vector3f &d) const {
    assert(abs(Norm(m_ray.m_d) - 1) < 1e-4);  // ray is initialized
    auto o = OffsetRayOrigin(m_p, m_ng, d);
    return {o, Normalize(d)};
  }
  virtual Ray SpawnRayTo(const Vector3f &p) const {
    assert(abs(Norm(m_ray.m_d) - 1) < 1e-4);  // ray is initialized
    Float      norm = (p - m_p).norm();
    auto       d    = (p - m_p) / norm;
    const auto o    = OffsetRayOrigin(m_p, m_ng, d);
    return {o, d, norm - SHADOW_EPS};
  }
  virtual Ray SpawnRayTo(const Interaction &it) const {
    return SpawnRayTo(it.m_p);
  }
  virtual bool isSInteraction() const { return false; }

  // shading normal, geometry normal
  Vector3f m_p, m_ns{Vector3f(0.0)}, m_ng{Vector3f(0.0)}, m_wo;
  // The ray that leads to this interaction
  Ray m_ray;

private:
};

class SInteraction : public Interaction {
public:
  SInteraction() = default;
  SInteraction(const Vector3f &p, const Vector3f &ns, const Vector3f &ng,
               const Vector3f &wo)
      : Interaction(p, ns, ng, wo) {}
  virtual ~SInteraction() = default;
  virtual Vector3f Le(const Vector3f &w) const;
  bool             isSInteraction() const override { return true; }
  Ray              SpawnRay(const Vector3f &d) const override;
  Ray              SpawnRayTo(const Vector3f &p) const override;
  Ray              SpawnRayTo(const Interaction &it) const override;

  const Primitive *m_primitive{nullptr};
};

class VInteraction : public Interaction {
public:
  VInteraction() = default;
  VInteraction(const Vector3f &p, const Vector3f &wo, Float g)
      : Interaction(p, Vector3f(0.0), Vector3f(0.0), wo), m_g(g) {}
  ~VInteraction() override = default;
  bool isSInteraction() const override { return false; }
  Ray  SpawnRay(const Vector3f &d) const override;
  Ray  SpawnRayTo(const Vector3f &p) const override;
  Ray  SpawnRayTo(const Interaction &it) const override;

  Float m_g{0};
};

FLG_NAMESPACE_END

#endif
