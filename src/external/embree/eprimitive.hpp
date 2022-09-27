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
  EmbreeMeshPrimitive(TriangleMesh *mesh, Resource &resource,
                      MaterialDispatcher *material = nullptr,
                      AreaLight *areaLight = nullptr, Volume *volume = nullptr);
  EmbreeMeshPrimitive(const std::string &path, Resource &resource,
                      MaterialDispatcher *material = nullptr,
                      AreaLight *areaLight = nullptr, Volume *volume = nullptr);
  ~EmbreeMeshPrimitive() override;

  // get the AABB bounding box of the primitive
  AABB getBound() const override;
  bool intersect(const Ray &ray, SInteraction &isect) const override;
  // if the areaLight actually exists
  AreaLight          *getAreaLight() const override;
  MaterialDispatcher *getMaterial() const override;
  Volume             *getVolume() const override { return m_volume; }

private:
  // initialized first
  Resource           *m_resource;  // pointer to resource manager
  TriangleMesh       *m_mesh;
  MaterialDispatcher *m_material{nullptr};
  AreaLight          *m_areaLight{nullptr};
  Volume             *m_volume{nullptr};

  // embree properties
  RTCDevice    m_device;
  RTCScene     m_scene;
  RTCGeometry  m_geom;
  unsigned int m_geomID;
};

FLG_NAMESPACE_END

#endif
