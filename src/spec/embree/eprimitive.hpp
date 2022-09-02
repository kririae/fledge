#ifndef __EPRIMITIVE_HPP__
#define __EPRIMITIVE_HPP__

#include <embree3/rtcore.h>
#include <embree3/rtcore_geometry.h>

#include <cstddef>
#include <memory>

#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"
#include "light.hpp"
#include "primitive.hpp"
#include "resource.hpp"

FLG_NAMESPACE_BEGIN

class EmbreeMeshPrimitive : public Primitive {
public:
  EmbreeMeshPrimitive(TriangleMesh *mesh, Material *material = nullptr,
                      AreaLight *areaLight = nullptr);
  EmbreeMeshPrimitive(const std::string &path, Material *material = nullptr,
                      AreaLight *areaLight = nullptr);
  ~EmbreeMeshPrimitive() override = default;

  // get the AABB bounding box of the primitive
  AABB getBound() const override;
  bool intersect(const Ray &ray, SInteraction &isect) const override;
  // if the areaLight actually exists
  AreaLight *getAreaLight() const override;
  Material  *getMaterial() const override;

private:
  TriangleMesh *m_mesh;
  Material     *m_material;
  AreaLight    *m_areaLight;

  // embree properties
  RTCDevice    m_device;
  RTCScene     m_scene;
  RTCGeometry  m_geom;
  unsigned int m_geomID;
  Resource     m_resource;
};

FLG_NAMESPACE_END

#endif
