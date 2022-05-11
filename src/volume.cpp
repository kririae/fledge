#include "volume.hpp"

#include <openvdb/Grid.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>

#include <memory>

#include "fwd.hpp"

SV_NAMESPACE_BEGIN

VDBVolume::VDBVolume(const std::string &filename) {
  // m_sigma_s and m_sigma_a are decided in advance
  m_sigma_t = m_sigma_a + m_sigma_s;

  openvdb::io::File file(filename);
  Log("load OpenVDB file %s", filename.c_str());
  file.open();
  // the only grid we need from cloud volume
  auto baseGrid = file.readGrid("density");
  m_grid        = std::shared_ptr<openvdb::FloatGrid>(
      openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid));
  Float minDensity, maxDensity;
  m_grid->evalMinMax(minDensity, maxDensity);
  Log("m_maxDensity=%f", maxDensity);
  m_maxDensity = Vector3f::Constant(maxDensity);
  auto aabb    = m_grid->evalActiveVoxelBoundingBox();
  // auto l_min = aabb.min(), l_max = aabb.max();

  file.close();
}

Vector3f VDBVolume::sample(const Ray &ray, Random &rng,
                           VInteraction &vi) const {
  auto accessor = m_grid->getConstAccessor();

  return Vector3f::Zero();
}

SV_NAMESPACE_END
