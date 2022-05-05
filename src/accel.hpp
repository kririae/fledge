#ifndef __ACCEL_HPP__
#define __ACCEL_HPP__

#include <memory>
#include <vector>

#include "fwd.hpp"
#include "interaction.hpp"
#include "primitive.hpp"
#include "shape.hpp"

SV_NAMESPACE_BEGIN

class Accel : public Primitive {
public:
  // must have an implementation.. whatever
  ~Accel() override = default;

  // derived from primitive.hpp
  bool intersect(const Ray &ray, SInteraction &isect) const override = 0;
  // return the scene's boundary AABB
  // TODO: to be implemented in Primitive
  virtual bool getAABB() = 0;
};

class NaiveAccel : public Accel {
public:
  NaiveAccel(const std::vector<std::shared_ptr<Primitive>> &p)
      : m_primitives(p) {}
  ~NaiveAccel() override = default;

  bool intersect(const Ray &ray, SInteraction &isect) const override {
    bool         res = false;
    SInteraction t_isect;
    for (auto &i : m_primitives) {
      if (i->intersect(ray, t_isect)) {
        res = true;
        if (t_isect.m_t < isect.m_t) isect = t_isect;
      }
    }

    return res;
  }

  bool getAABB() override {
    TODO();
    return false;
  }

  std::vector<std::shared_ptr<Primitive>> m_primitives;
};

SV_NAMESPACE_END

#endif
