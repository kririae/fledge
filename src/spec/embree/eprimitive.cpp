#include "eprimitive.hpp"

#include <embree3/rtcore_buffer.h>
#include <embree3/rtcore_common.h>
#include <embree3/rtcore_device.h>
#include <embree3/rtcore_geometry.h>
#include <embree3/rtcore_scene.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <type_traits>

#include "common/aabb.h"
#include "common/vector.h"
#include "debug.hpp"
#include "embree.hpp"
#include "fledge.h"
#include "plymesh.hpp"
#include "shape.hpp"

FLG_NAMESPACE_BEGIN

EmbreeMeshPrimitive::EmbreeMeshPrimitive(TriangleMesh *mesh, Resource &resource,
                                         Material  *material,
                                         AreaLight *areaLight, Volume *volume)
    : m_resource(&resource),
      m_mesh(mesh),
      m_material(material),
      m_areaLight(areaLight),
      m_volume(volume) {
  m_device = embreeInitializeDevice();
  m_scene  = rtcNewScene(m_device);
  m_geom   = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);

  static_assert(std::is_same_v<Float, float>,
                "currently only Float=float is supported");

  // C-style initialization
  float *verticies = (float *)rtcSetNewGeometryBuffer(
      m_geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float),
      m_mesh->nVert);
  unsigned int *indicies = (unsigned int *)rtcSetNewGeometryBuffer(
      m_geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
      3 * sizeof(unsigned int), m_mesh->nInd / 3);
  float *normals;
  if (m_mesh->n != nullptr) {
    // Added normal data
    rtcSetGeometryVertexAttributeCount(m_geom, 1);
    normals = (float *)rtcSetNewGeometryBuffer(
        m_geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, RTC_FORMAT_FLOAT3,
        3 * sizeof(float), m_mesh->nVert);
    C(normals);
    memcpy(normals, m_mesh->n, 3 * sizeof(float) * m_mesh->nVert);
  }

  C(verticies, indicies);

  memcpy(verticies, m_mesh->p, 3 * sizeof(float) * m_mesh->nVert);
  // Notice the implicit conversion
  memcpy(indicies, m_mesh->ind, sizeof(int) * m_mesh->nInd);

  rtcSetGeometryBuildQuality(m_geom, RTC_BUILD_QUALITY_HIGH);
  rtcCommitGeometry(m_geom);
  m_geomID = rtcAttachGeometry(m_scene, m_geom);
  rtcReleaseGeometry(m_geom);
  rtcCommitScene(m_scene);
  // Embree init finished
  SLog("Embree is ready");
}

EmbreeMeshPrimitive::EmbreeMeshPrimitive(const std::string &path,
                                         Resource &resource, Material *material,
                                         AreaLight *areaLight, Volume *volume)
    : EmbreeMeshPrimitive(MakeTriangleMesh(path, resource), resource, material,
                          areaLight, volume) {}

EmbreeMeshPrimitive::~EmbreeMeshPrimitive() {
  rtcReleaseScene(m_scene);
}

AABB EmbreeMeshPrimitive::getBound() const {
  // > The provided destination pointer must be aligned to 16 bytes.
  // This allocation do not use resource
  auto *bounds = (RTCBounds *)std::aligned_alloc(16, sizeof(RTCBounds));
  rtcGetSceneBounds(m_scene, bounds);
  AABB res(Vector3f{bounds->lower_x, bounds->lower_y, bounds->lower_z},
           Vector3f{bounds->upper_x, bounds->upper_y, bounds->upper_z});
  std::free(bounds);
  return res;
}

bool EmbreeMeshPrimitive::intersect(const Ray &ray, SInteraction &isect) const {
  RTCIntersectContext context;
  rtcInitIntersectContext(&context);

  RTCRayHit rayhit;
  rayhit.ray.org_x     = ray.m_o.x();
  rayhit.ray.org_y     = ray.m_o.y();
  rayhit.ray.org_z     = ray.m_o.z();
  rayhit.ray.dir_x     = ray.m_d.x();
  rayhit.ray.dir_y     = ray.m_d.y();
  rayhit.ray.dir_z     = ray.m_d.z();
  rayhit.ray.tnear     = 0;
  rayhit.ray.tfar      = ray.m_tMax;
  rayhit.ray.mask      = -1;
  rayhit.ray.flags     = 0;
  rayhit.hit.geomID    = RTC_INVALID_GEOMETRY_ID;
  rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

  rtcIntersect1(m_scene, &context, &rayhit);

  if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
    assert(rayhit.hit.geomID == m_geomID);
    assert(rayhit.ray.tfar > 0);
    ray.m_tMax = rayhit.ray.tfar;
    isect.m_p  = ray(ray.m_tMax);
    isect.m_wo = -ray.m_d;
    isect.m_ng =
        Normalize(Vector3f{rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z});
    // interpolate through
    isect.m_ns = isect.m_ng;
    if (m_mesh->n) {
#if 1
      rtcInterpolate0(m_geom, 0, rayhit.hit.u, rayhit.hit.v,
                      RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, isect.m_ns.m_vec, 0);
      NormalizeInplace(isect.m_ns);
#endif
    }

    isect.m_primitive = this;
    isect.m_ray       = ray;
    return true;
  } else {
    return false;
  }
}

AreaLight *EmbreeMeshPrimitive::getAreaLight() const {
  return m_areaLight;
}

Material *EmbreeMeshPrimitive::getMaterial() const {
  return m_material;
}

FLG_NAMESPACE_END
