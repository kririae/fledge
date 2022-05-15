#ifndef __PRIMITIVE_HPP__
#define __PRIMITIVE_HPP__

#include <memory>

#include "fwd.hpp"
#include "material.hpp"

SV_NAMESPACE_BEGIN

class AreaLight;

class Primitive {
public:
  virtual ~Primitive() = default;

  // will modify the mutable ray.tMax
  virtual bool intersect(const Ray &ray, SInteraction &isect) const = 0;

  // get the AABB bounding box of the primitive
  virtual AABB getBound() const = 0;

  // getter
  virtual AreaLight *getAreaLight() const = 0;
  virtual Material  *getMaterial() const  = 0;
};

class ShapePrimitive : public Primitive {
public:
  // If the primitive is a areaLight, areaLight.m_shape must be
  // the shape passing to the ctor
  ShapePrimitive(
      const std::shared_ptr<Shape>    &shape,
      const std::shared_ptr<Material> &material =  // default to diffuse
      std::make_shared<DiffuseMaterial>(Vector3f::Ones()),
      const std::shared_ptr<AreaLight> &areaLight = nullptr);
  ~ShapePrimitive() override = default;

  // get the AABB bounding box of the primitive
  AABB getBound() const override;
  bool intersect(const Ray &ray, SInteraction &isect) const override;
  // if the areaLight actually exists
  AreaLight *getAreaLight() const override;
  Material  *getMaterial() const override;

private:
  std::shared_ptr<Shape>     m_shape;
  std::shared_ptr<Material>  m_material;
  std::shared_ptr<AreaLight> m_areaLight;
};

// The Accel is derived from Primitive
// see accel.hpp

SV_NAMESPACE_END

#endif
