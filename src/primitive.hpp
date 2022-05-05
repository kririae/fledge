#ifndef __PRIMITIVE_HPP__
#define __PRIMITIVE_HPP__

#include <memory>

#include "fwd.hpp"
#include "interaction.hpp"
#include "shape.hpp"

SV_NAMESPACE_BEGIN

class AreaLight;

class Primitive {
public:
  virtual ~Primitive() = default;

  // will modify the mutable ray.tMax
  virtual bool       intersect(const Ray &ray, SInteraction &isect) const = 0;
  virtual AreaLight *getAreaLight() const                                 = 0;
};

class ShapePrimitive : public Primitive {
public:
  // If the primitive is a areaLight, areaLight.m_shape must be
  // the shape passing to the ctor
  ShapePrimitive(const std::shared_ptr<Shape>     &shape,
                 const std::shared_ptr<AreaLight> &areaLight = nullptr);
  ~ShapePrimitive() override = default;
  bool intersect(const Ray &ray, SInteraction &isect) const override;
  // if the areaLight actually exists
  AreaLight *getAreaLight() const override;

private:
  std::shared_ptr<Shape>     m_shape;
  std::shared_ptr<AreaLight> m_areaLight;
};

// The Accel is derived from Primitive
// see accel.hpp

SV_NAMESPACE_END

#endif
