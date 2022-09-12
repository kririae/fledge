#include "bvh.hpp"

#include <algorithm>
#include <cstring>
#include <memory_resource>

#include "debug.hpp"
#include "fledge.h"
#include "spec/embree/embree.hpp"

FLG_NAMESPACE_BEGIN

namespace experimental {
void BasicBVHBuilder::build() {
  int       n_triangles = m_mesh->nInd / 3;
  Triangle *triangles =
      (Triangle *)m_resource.allocate(sizeof(Triangle) * n_triangles);
  for (int i = 0; i < n_triangles; ++i) {
    triangles[i].m_mesh = m_mesh;
    triangles[i].m_v    = m_mesh->ind + i * 3;
  }  // for
  m_root = recursiveBuilder(triangles, n_triangles, 0);
}
bool BasicBVHBuilder::intersect(BVHRayHit &rayhit) {
  TODO();
  return true;
}

BasicBVHBuilder::BasicBVHNode *BasicBVHBuilder::recursiveBuilder(
    Triangle *triangles, std::size_t n_triangles, int depth) {
  if (triangles == nullptr || n_triangles == 0) return nullptr;

  // Allocate the node here
  BasicBVHNode *node =
      std::pmr::polymorphic_allocator<BasicBVHNode>{&m_resource}
          .new_object<BasicBVHNode>();
  if (n_triangles <= 8 || depth >= 26) {
    BVHBound bound = triangles[0].getBound();
    for (std::size_t i = 1; i < n_triangles; ++i)
      bound.merge(triangles[i].getBound());
    node->bound       = bound;
    node->triangles   = triangles;
    node->n_triangles = n_triangles;
    return node;
  }  // if (n_triangles <= 8 || depth >= 26) {

  int dim = depth % 3;  // the partition dimension

  Triangle *mid_triangles = triangles + n_triangles / 2;
  // clang-format off
  // TODO: add execution policy
  std::nth_element(triangles, mid_triangles, triangles + n_triangles,
  [dim](const Triangle &a, const Triangle &b) -> bool {
    switch (dim) {
      case 0:
        return a.centerX3() < b.centerX3();
      case 1:
        return a.centerY3() < b.centerY3();
      case 2:
        return a.centerZ3() < b.centerZ3();
      default:
        return false;
    };
  }); // std::nth_element

  Float mid = mid_triangles->center3()[dim];
  // TODO: add execution policy
  mid_triangles = std::partition(triangles, triangles + n_triangles,
  [mid, dim](const Triangle &a) -> bool {
    switch (dim) {
      case 0:
        return a.centerX3() < mid;
      case 1:
        return a.centerY3() < mid;
      case 2:
        return a.centerZ3() < mid;
      default:
        return false;
    };
  }); // std::partition
  // clang-format on

  node->left =
      recursiveBuilder(triangles, mid_triangles - triangles, depth + 1);
  node->right = recursiveBuilder(
      mid_triangles, triangles + n_triangles - mid_triangles, depth + 1);
  // Some spj
  if (node->left != nullptr) node->bound.merge(node->left->bound);
  if (node->right != nullptr) node->bound.merge(node->right->bound);
  if (node->left == nullptr && node->right == nullptr) return nullptr;
  return node;
}

RefBVHBuilder::RefBVHBuilder(InternalTriangleMesh      *mesh,
                             std::pmr::memory_resource *mem_resource)
    : BVHBuilderBase(mesh, mem_resource) {
  // Currently, code adopted from eprimitive.cpp, since the interface is almost
  // the same
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

  memcpy(verticies, m_mesh->p, 3 * sizeof(float) * m_mesh->nVert);
  // Notice the implicit conversion
  memcpy(indicies, m_mesh->ind, sizeof(int) * m_mesh->nInd);
}

void RefBVHBuilder::build() {
  rtcSetGeometryBuildQuality(m_geom, RTC_BUILD_QUALITY_HIGH);
  rtcCommitGeometry(m_geom);
  m_geomID = rtcAttachGeometry(m_scene, m_geom);
  rtcReleaseGeometry(m_geom);
  rtcCommitScene(m_scene);  // start building
}

RefBVHBuilder::BVHBound RefBVHBuilder::getBound() const {
  auto *bounds = (RTCBounds *)std::aligned_alloc(16, sizeof(RTCBounds));
  rtcGetSceneBounds(m_scene, bounds);
  BVHBound res{
      .lower = {bounds->lower_x, bounds->lower_y, bounds->lower_z},
      .upper = {bounds->upper_x, bounds->upper_y, bounds->upper_z}
  };
  std::free(bounds);
  return res;
}

bool RefBVHBuilder::intersect(BVHRayHit &rayhit) {
  RTCIntersectContext context;
  rtcInitIntersectContext(&context);

  RTCRayHit r_rayhit;
  r_rayhit.ray.org_x     = rayhit.ray_o.x();
  r_rayhit.ray.org_y     = rayhit.ray_o.y();
  r_rayhit.ray.org_z     = rayhit.ray_o.z();
  r_rayhit.ray.dir_x     = rayhit.ray_d.x();
  r_rayhit.ray.dir_y     = rayhit.ray_d.y();
  r_rayhit.ray.dir_z     = rayhit.ray_d.z();
  r_rayhit.ray.tnear     = rayhit.tnear;
  r_rayhit.ray.tfar      = rayhit.tfar;
  r_rayhit.ray.mask      = -1;
  r_rayhit.ray.flags     = 0;
  r_rayhit.hit.geomID    = RTC_INVALID_GEOMETRY_ID;
  r_rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

  rtcIntersect1(m_scene, &context, &r_rayhit);

  if (r_rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
    rayhit.tfar   = r_rayhit.ray.tfar;
    rayhit.hit_ng = rayhit.hit_ns = Normalize(
        Vector3f{r_rayhit.hit.Ng_x, r_rayhit.hit.Ng_y, r_rayhit.hit.Ng_z});
    rayhit.hit = true;
    return true;
  } else {
    return false;
  }
}
}  // namespace experimental

FLG_NAMESPACE_END