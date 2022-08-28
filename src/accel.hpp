#ifndef __ACCEL_HPP__
#define __ACCEL_HPP__

#include <memory>
#include <vector>

#include "common/aabb.h"
#include "fwd.hpp"
#include "primitive.hpp"
#include "common/vector.h"

FLG_NAMESPACE_BEGIN

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

protected:
  std::vector<std::shared_ptr<Primitive>> m_primitives;
};

class NaiveBVHAccel : public Accel {
  // Naive BVH implementation without extensive optimization
  // Recursively defined

public:
  NaiveBVHAccel(std::vector<std::shared_ptr<Primitive>> p, int depth = 0,
                AABB *box = nullptr);
  ~NaiveBVHAccel() override = default;
  bool   intersect(const Ray &ray, SInteraction &isect) const override;
  AABB   getBound() const override;
  size_t getMemoryUsage() const { return m_memory_usage; }
  size_t getDepth() const { return m_depth; }

protected:
  AABB                           m_box;
  std::shared_ptr<NaiveBVHAccel> m_left, m_right;

  std::vector<std::shared_ptr<Primitive>> m_primitives;

  size_t m_memory_usage, m_depth;
};

FLG_NAMESPACE_END

#endif
