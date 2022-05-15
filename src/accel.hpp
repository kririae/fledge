#ifndef __ACCEL_HPP__
#define __ACCEL_HPP__

#include <memory>
#include <vector>

#include "fwd.hpp"
#include "primitive.hpp"

SV_NAMESPACE_BEGIN

class Accel : public Primitive {
public:
  // must have an implementation.. whatever
  ~Accel() override = default;

  // derived from primitive.hpp
  bool       intersect(const Ray &ray, SInteraction &isect) const override = 0;
  AreaLight *getAreaLight() const override { return nullptr; }
  Material  *getMaterial() const override { return nullptr; }
};

class NaiveAccel : public Accel {
public:
  NaiveAccel(const std::vector<std::shared_ptr<Primitive>> &p)
      : m_primitives(p) {}
  ~NaiveAccel() override = default;
  bool intersect(const Ray &ray, SInteraction &isect) const override;
  AABB getBound() const override;

private:
  std::vector<std::shared_ptr<Primitive>> m_primitives;
};

SV_NAMESPACE_END

#endif
