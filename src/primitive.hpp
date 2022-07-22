#ifndef __PRIMITIVE_HPP__
#define __PRIMITIVE_HPP__

#include <memory>

#include "fwd.hpp"
#include "material.hpp"

SV_NAMESPACE_BEGIN

class AreaLight;

// > The abstract Primitive base class is the bridge between the geometry
// processing and shading subsystems of pbrt.
// > However, not exactly the same in our render
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

class MeshPrimitive : public Primitive {
  // To construct a triangle mesh, you could of course use a bunch of
  // ShapePrimitive s, assigning each triangle with Material.
  // However, we provide a MethPrimitive with a higher level of abstraction

public:
  MeshPrimitive(
      const std::string               &path,
      const std::shared_ptr<Material> &material =  // default to diffuse
      std::make_shared<DiffuseMaterial>(Vector3f::Ones()),
      const std::shared_ptr<AreaLight> &areaLight = nullptr);
  ~MeshPrimitive() override = default;

  // get the AABB bounding box of the primitive
  AABB getBound() const override;
  bool intersect(const Ray &ray, SInteraction &isect) const override;
  // if the areaLight actually exists
  AreaLight *getAreaLight() const override;
  Material  *getMaterial() const override;

private:
  // Mesh-related
  std::shared_ptr<TriangleMesh>          m_mesh;
  std::vector<std::shared_ptr<Triangle>> m_triangles;
  std::shared_ptr<Accel>                 m_accel;

  std::shared_ptr<Material>  m_material;
  std::shared_ptr<AreaLight> m_areaLight;
};

// The Accel is derived from Primitive
// see accel.hpp
SV_NAMESPACE_END

#endif
