#ifndef __PRIMITIVE_HPP__
#define __PRIMITIVE_HPP__

#include <memory>

#include "common/transform.h"
#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"
#include "material.hpp"
#include "resource.hpp"

FLG_NAMESPACE_BEGIN

class AreaLight;

/**
 * @brief This Primitive base class is the bridge between the geometry
 * processing and shading subsystems.
 * @note There are two types of Primitive implementation, one is the *real*
 * Primitive implementation, which strictly implements the interface and
 * generally will not evaluate recursively, i.e., dependents on other primitive
 * classes. The other is the *pseudo* primitive class, which behaves seemingly
 * the same as the original Primitive class, but actually it is the aggregation
 * of primitive classes.
 */
class Primitive {
public:
  Primitive()          = default;
  virtual ~Primitive() = default;

  /**
   * @brief Accept a ray and consider its origin, direction and m_tMax to
   * perform intersection with the object contained in the class. It will modify
   * the intersection and ray.m_tMax iff there's intersection. Else the
   * parameters are retained.
   * @see Scene.intersect(...)
   *
   * @param ray
   * @param isect
   * @return true The function will return true iff there's an intersection.
   */
  virtual bool intersect(const Ray &ray, SInteraction &isect) const = 0;

  /**
   * @brief Get the boundary of the primitive
   *
   * @return AABB The shape AABB representing the boundary.
   */
  virtual AABB getBound() const = 0;

  /**
   * The followings are getters for the class.
   */
  /**
   * @brief Get the Material of the primitive
   * @note Persistence of the object is guaranteed.
   * @return Material*
   */
  virtual Material *getMaterial() const = 0;
  /**
   * @brief Get the AreaLight of the primitive if it really exists
   * @note Persistence of the object is guaranteed.
   * @return AreaLight*
   */
  virtual AreaLight *getAreaLight() const { return nullptr; }
  /**
   * @brief Get the Volume of the primitive if is really exists
   * @note Persistence of the object is guaranteed.
   * @return Volume*
   */
  virtual Volume *getVolume() const { return nullptr; }
};

class ShapePrimitive : public Primitive {
public:
  // If the primitive is a areaLight, areaLight.m_shape must be
  // the shape passing to the ctor
  ShapePrimitive(Shape *shape, Material *material = nullptr,
                 AreaLight *areaLight = nullptr, Volume *volume = nullptr);
  ~ShapePrimitive() override = default;

  // get the AABB bounding box of the primitive
  AABB getBound() const override;
  bool intersect(const Ray &ray, SInteraction &isect) const override;
  // if the areaLight actually exists
  AreaLight *getAreaLight() const override;
  Material  *getMaterial() const override;
  // if the volume actually exists, the result will not be nullptr
  Volume *getVolume() const override;

private:
  Shape     *m_shape;
  Material  *m_material;
  AreaLight *m_areaLight;
  Volume    *m_volume;
};

class MeshPrimitive : public Primitive {
  // To construct a triangle mesh, you could of course use a bunch of
  // ShapePrimitive s, assigning each triangle with Material.
  // However, we provide a MethPrimitive with a higher level of abstraction

public:
  MeshPrimitive(TriangleMesh *mesh, Resource &resource,
                Material *material = nullptr, AreaLight *areaLight = nullptr);
  MeshPrimitive(const std::string &path, Resource &resource,
                Material *material = nullptr, AreaLight *areaLight = nullptr);
  ~MeshPrimitive() override = default;

  // get the AABB bounding box of the primitive
  AABB getBound() const override;
  bool intersect(const Ray &ray, SInteraction &isect) const override;
  // if the areaLight actually exists
  AreaLight *getAreaLight() const override;
  Material  *getMaterial() const override;

private:
  // Mesh-related
  Resource               *m_resource;
  TriangleMesh           *m_mesh;
  std::vector<Triangle *> m_triangles;
  Accel                  *m_accel;

  Material  *m_material;
  AreaLight *m_areaLight;
};

// class VolumePremitive: public

// The Accel is derived from Primitive
// see accel.hpp
FLG_NAMESPACE_END

#endif
