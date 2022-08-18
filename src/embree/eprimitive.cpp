#include "eprimitive.hpp"

#include <embree3/rtcore_buffer.h>
#include <embree3/rtcore_common.h>
#include <embree3/rtcore_geometry.h>
#include <embree3/rtcore_scene.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <type_traits>

#include "aabb.hpp"
#include "debug.hpp"
#include "embree.hpp"
#include "fwd.hpp"
#include "plymesh.hpp"
#include "vector.hpp"

SV_NAMESPACE_BEGIN

EmbreeMeshPrimitive::EmbreeMeshPrimitive(
    const std::shared_ptr<TriangleMesh> &mesh,
    const std::shared_ptr<Material>     &material,
    const std::shared_ptr<AreaLight>    &areaLight)
    : m_mesh(mesh), m_material(material), m_areaLight(areaLight) {
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
  C(verticies);
  C(indicies);

  memcpy(verticies, m_mesh->p.get(), 3 * sizeof(float) * m_mesh->nVert);
  // Notice the implicit conversion
  memcpy(indicies, m_mesh->ind.get(), sizeof(int) * m_mesh->nInd);

  rtcSetGeometryBuildQuality(m_geom, RTC_BUILD_QUALITY_HIGH);
  rtcCommitGeometry(m_geom);
  m_geomID = rtcAttachGeometry(m_scene, m_geom);
  rtcReleaseGeometry(m_geom);
  rtcCommitScene(m_scene);
  // Embree init finished
  SLog("Embree is ready");
}

EmbreeMeshPrimitive::EmbreeMeshPrimitive(
    const std::string &path, const std::shared_ptr<Material> &material,
    const std::shared_ptr<AreaLight> &areaLight)
    : EmbreeMeshPrimitive(MakeTriangleMesh(path), material, areaLight) {}

AABB EmbreeMeshPrimitive::getBound() const {
  // > The provided destination pointer must be aligned to 16 bytes.
  auto *bounds = (RTCBounds *)std::aligned_alloc(16, sizeof(RTCBounds));
  rtcGetSceneBounds(m_scene, bounds);
  AABB res(Vector3f{bounds->lower_x, bounds->lower_y, bounds->lower_z},
           Vector3f{bounds->upper_x, bounds->upper_y, bounds->upper_z});
  std::free(bounds);
  return res;
}

bool EmbreeMeshPrimitive::intersect(const Ray &ray, SInteraction &isect) const {
  // TODO: should it be kept outside?
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
    isect.m_ng = Vector3f{rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z}
                     .stableNormalized();
    if (isect.m_ng.dot(ray.m_d) > 0) isect.m_ng = -isect.m_ng;
    isect.m_ns        = isect.m_ng;
    isect.m_primitive = this;
    return true;
  } else {
    return false;
  }
}

AreaLight *EmbreeMeshPrimitive::getAreaLight() const {
  return m_areaLight.get();
}

Material *EmbreeMeshPrimitive::getMaterial() const {
  return m_material.get();
}

SV_NAMESPACE_END
