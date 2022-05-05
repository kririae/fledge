#ifndef __PRIMITIVE_HPP__
#define __PRIMITIVE_HPP__

#include <memory>

#include "fwd.hpp"
#include "interaction.hpp"
#include "shape.hpp"

SV_NAMESPACE_BEGIN

class Primitive {
public:
  virtual ~Primitive() = default;

  virtual bool intersect(const Ray &ray, SInteraction &isect) const = 0;
};

class ShapePrimitive : public Primitive {
public:
  ShapePrimitive(const std::shared_ptr<Shape> &shape) : m_shape(shape) {}
  ~ShapePrimitive() override = default;
  bool intersect(const Ray &ray, SInteraction &isect) const override {
    return m_shape->intersect(ray, isect);
  }

private:
  std::shared_ptr<Shape> m_shape;
};

// The Accel is derived from Primitive
// see accel.hpp

SV_NAMESPACE_END

#endif
