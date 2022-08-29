#ifndef __EPRIMITIVE_HPP__
#define __EPRIMITIVE_HPP__

#include <embree3/rtcore.h>
#include <embree3/rtcore_geometry.h>

#include <memory>

#include "fledge.h"
#include "debug.hpp"
#include "light.hpp"
#include "primitive.hpp"
#include "common/vector.h"

FLG_NAMESPACE_BEGIN

class EmbreeMeshPrimitive : public Primitive {
public:
  EmbreeMeshPrimitive(
      const std::shared_ptr<TriangleMesh> &mesh,
      const std::shared_ptr<Material>     &material =  // default to diffuse
      std::make_shared<DiffuseMaterial>(Vector3f(1.0)),
      const std::shared_ptr<AreaLight> &areaLight = nullptr);
  EmbreeMeshPrimitive(
      const std::string               &path,
      const std::shared_ptr<Material> &material =  // default to diffuse
      std::make_shared<DiffuseMaterial>(Vector3f(1.0)),
      const std::shared_ptr<AreaLight> &areaLight = nullptr);
  ~EmbreeMeshPrimitive() override = default;

  // get the AABB bounding box of the primitive
  AABB getBound() const override;
  bool intersect(const Ray &ray, SInteraction &isect) const override;
  // if the areaLight actually exists
  AreaLight *getAreaLight() const override;
  Material  *getMaterial() const override;

private:
  std::shared_ptr<TriangleMesh> m_mesh;
  std::shared_ptr<Material>     m_material;
  std::shared_ptr<AreaLight>    m_areaLight;

  // embree properties
  RTCDevice    m_device;
  RTCScene     m_scene;
  RTCGeometry  m_geom;
  unsigned int m_geomID;
};

FLG_NAMESPACE_END

#endif
