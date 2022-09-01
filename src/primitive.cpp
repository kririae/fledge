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

ShapePrimitive::ShapePrimitive(const std::shared_ptr<Shape>     &shape,
                               const std::shared_ptr<Material>  &material,
                               const std::shared_ptr<AreaLight> &areaLight,
                               const std::shared_ptr<Volume>    &volume)
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
  return m_areaLight.get();
}

Material *ShapePrimitive::getMaterial() const {
  return m_material.get();
}

Volume *ShapePrimitive::getVolume() const {
  // if the volume exists, the result is not nullptr
  return m_volume.get();
}

MeshPrimitive::MeshPrimitive(const std::shared_ptr<TriangleMesh> &mesh,
                             const std::shared_ptr<Material>     &material,
                             const std::shared_ptr<AreaLight>    &areaLight)
    : m_mesh(mesh), m_material(material), m_areaLight(areaLight) {
  assert(m_mesh->nInd % 3 == 0);
  for (int i = 0; i < m_mesh->nInd / 3; ++i)
    m_triangles.push_back(std::make_shared<Triangle>(m_mesh, i));

  // AND, construct the BVH under the mesh
  // Convert triangles to primitive
  std::vector<std::shared_ptr<Primitive>> p_triangles;
  for (size_t i = 0; i < m_triangles.size(); ++i)
    p_triangles.push_back(
        std::make_shared<ShapePrimitive>(ShapePrimitive(m_triangles[i])));
  m_accel = std::shared_ptr<NaiveBVHAccel>(new NaiveBVHAccel(p_triangles));
}

MeshPrimitive::MeshPrimitive(const std::string                &path,
                             const std::shared_ptr<Material>  &material,
                             const std::shared_ptr<AreaLight> &areaLight)
    : MeshPrimitive(MakeTriangleMesh(path), material, areaLight) {}

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
  return m_areaLight.get();
}

Material *MeshPrimitive::getMaterial() const {
  return m_material.get();
}

FLG_NAMESPACE_END
