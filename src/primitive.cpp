#include "primitive.hpp"

#include <filesystem>
#include <memory>

#include "accel.hpp"
#include "common/aabb.h"
#include "common/ray.h"
#include "interaction.hpp"
#include "light.hpp"
#include "plymesh.hpp"
#include "shape.hpp"
#include "volume.hpp"

FLG_NAMESPACE_BEGIN

ShapePrimitive::ShapePrimitive(Shape *shape, Material *material,
                               AreaLight *areaLight, Volume *volume)
    : m_shape(shape),
      m_material(material),
      m_areaLight(areaLight),
      m_volume(volume) {
  if (volume != nullptr) m_volume->setBound(getBound());
}

AABB ShapePrimitive::getBound() const {
  return m_shape->getBound();
}

bool ShapePrimitive::intersect(const Ray &ray, SInteraction &isect) const {
  Float tHit;
  if (!m_shape->intersect(ray, tHit, isect)) return false;

  ray.m_tMax        = tHit;
  isect.m_primitive = this;
  isect.m_ray       = ray;
  return true;
}

// if the areaLight actually exists
AreaLight *ShapePrimitive::getAreaLight() const {
  return m_areaLight;
}

Material *ShapePrimitive::getMaterial() const {
  return m_material;
}

Volume *ShapePrimitive::getVolume() const {
  // if the volume exists, the result is not nullptr
  return m_volume;
}

MeshPrimitive::MeshPrimitive(TriangleMesh *mesh, Resource &resource,
                             Material *material, AreaLight *areaLight)
    : m_resource(&resource),
      m_mesh(mesh),
      m_material(material),
      m_areaLight(areaLight) {
  assert(m_mesh->nInd % 3 == 0);
  for (int i = 0; i < m_mesh->nInd / 3; ++i)
    m_triangles.push_back(m_resource->alloc<Triangle>(m_mesh, i));

  // AND, construct the BVH under the mesh
  // Convert triangles to primitive
  std::vector<Primitive *> p_triangles;
  for (size_t i = 0; i < m_triangles.size(); ++i)
    p_triangles.push_back(m_resource->alloc<ShapePrimitive>(m_triangles[i]));
  m_accel = m_resource->alloc<NaiveBVHAccel>(p_triangles, *m_resource);
}

MeshPrimitive::MeshPrimitive(const std::string &path, Resource &resource,
                             Material *material, AreaLight *areaLight)
    : MeshPrimitive(MakeTriangleMesh(path, resource), resource, material,
                    areaLight) {}

AABB MeshPrimitive::getBound() const {
  return m_accel->getBound();
}

bool MeshPrimitive::intersect(const Ray &ray, SInteraction &isect) const {
  // Accel implements the interface exactly the same as primitive, so no more
  // operation is needed
  if (!m_accel->intersect(ray, isect)) return false;

  isect.m_primitive = this;
  isect.m_ray       = ray;  // before transformation
  return true;
}

AreaLight *MeshPrimitive::getAreaLight() const {
  return m_areaLight;
}

Material *MeshPrimitive::getMaterial() const {
  return m_material;
}

FLG_NAMESPACE_END
