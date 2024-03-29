#ifndef __ACCEL_HPP__
#define __ACCEL_HPP__

#include <memory>
#include <vector>

#include "common/aabb.h"
#include "primitive.hpp"

FLG_NAMESPACE_BEGIN

class Accel : public Primitive {
public:
  // must have an implementation.. whatever
  ~Accel() override = default;

  // derived from primitive.hpp
  bool       intersect(const Ray &ray, SInteraction &isect) const override = 0;
  AreaLight *getAreaLight() const override { return nullptr; }
  MaterialDispatcher *getMaterial() const override { return nullptr; }
};

class NaiveAccel : public Accel {
public:
  NaiveAccel(const std::vector<Primitive *> &p) : m_primitives(p) {}
  ~NaiveAccel() override = default;
  bool intersect(const Ray &ray, SInteraction &isect) const override;
  AABB getBound() const override;

protected:
  std::vector<Primitive *> m_primitives;
};

class NaiveBVHAccel : public Accel {
  // Naive BVH implementation without extensive optimization
  // Recursively defined

public:
  NaiveBVHAccel(std::vector<Primitive *> p, Resource &resource, int depth = 0,
                AABB *box = nullptr);
  ~NaiveBVHAccel() override = default;
  bool   intersect(const Ray &ray, SInteraction &isect) const override;
  AABB   getBound() const override;
  size_t getMemoryUsage() const { return m_memory_usage; }
  size_t getDepth() const { return m_depth; }

protected:
  AABB           m_box;
  NaiveBVHAccel *m_left, *m_right;

  std::vector<Primitive *> m_primitives;

  size_t m_memory_usage, m_depth;
};

FLG_NAMESPACE_END

#endif
