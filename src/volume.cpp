#include "volume.hpp"

#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

#include <memory>

#include "aabb.hpp"
#include "fwd.hpp"
#include "interaction.hpp"

SV_NAMESPACE_BEGIN

VDBVolume::VDBVolume(const std::string &filename) {
  // m_sigma_s and m_sigma_a are decided in advance
  m_sigma_a = 0.1;
  m_sigma_s = 1.5;
  m_g       = -0.3;
  m_sigma_t = m_sigma_a + m_sigma_s;

  openvdb::initialize();
  openvdb::io::File file(filename);
  SV_Log("load OpenVDB file %s", filename.c_str());
  file.open();
  // the only grid we need from cloud volume
  auto baseGrid = file.readGrid("density");
  m_grid        = std::shared_ptr<openvdb::FloatGrid>(
      openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid));

  // acquire the global maxDensity in nodes
  Float minDensity, maxDensity;
  m_grid->evalMinMax(minDensity, maxDensity);
  SV_Log("m_maxDensity=%f", maxDensity);
  m_maxDensity    = maxDensity;
  m_invMaxDensity = 1.0 / m_maxDensity;

  auto aabb  = m_grid->evalActiveVoxelBoundingBox();
  auto l_min = aabb.min(), l_max = aabb.max();
  auto w_min = m_grid->indexToWorld(l_min);
  auto w_max = m_grid->indexToWorld(l_max);
  SV_Log("AABB low=(%f,%f,%f) max=(%f,%f,%f)", w_min.x(), w_min.y(), w_min.z(),
         w_max.x(), w_max.y(), w_max.z());

  // initialize here
  m_aabb = std::make_shared<AABB>(
      Vector3f{Float(w_min[0]), Float(w_min[1]), Float(w_min[2])},
      Vector3f{Float(w_max[0]), Float(w_max[1]), Float(w_max[2])});

  file.close();
}

Vector3f VDBVolume::tr(const Ray &ray, Random &rng) const {
  auto accessor      = m_grid->getConstAccessor();
  using sampler_type = openvdb::tools::Sampler<1>;
  // Extra effort is needed to use openvdb to interpolate
  // for simplicity, we currently will not use openvdb::ray
  auto sampler = openvdb::tools::GridSampler<decltype(accessor), sampler_type>(
      accessor, m_grid->transform());
  Float t_min, t_max;
  if (!m_aabb->intersect(ray, t_min, t_max)) return Vector3f::Ones();

  // ratio-tracking (I cannot understand it correctness currently)
  Float tr = 1, t = t_min;
  Float max_mu = m_sigma_t * m_maxDensity;
  while (true) {
    t += -std::log(1 - rng.get1D()) / max_mu;
    if (t > t_max) break;
    Vector3f pos  = ray(t);
    Float density = sampler.wsSample(openvdb::Vec3d{pos.x(), pos.y(), pos.z()});
    tr *= 1 - std::max(static_cast<Float>(0), density * m_invMaxDensity);
  }

  return Vector3f::Constant(tr);
}

Vector3f VDBVolume::sample(const Ray &ray, Random &rng,
                           VInteraction &vi) const {
  // Delta-Tracking
  auto accessor      = m_grid->getConstAccessor();
  using sampler_type = openvdb::tools::Sampler<1>;
  // Extra effort is needed to use openvdb to interpolate
  // for simplicity, we currently will not use openvdb::ray
  auto sampler = openvdb::tools::GridSampler<decltype(accessor), sampler_type>(
      accessor, m_grid->transform());
  Float t_min, t_max;
  if (!m_aabb->intersect(ray, t_min, t_max)) return Vector3f::Ones();

  Float t      = t_min;
  Float max_mu = m_sigma_t * m_maxDensity;
  while (true) {
    // free-path sampling
    // delta-tracking in \overline{\mu}
    t += -std::log(1 - rng.get1D()) / max_mu;
    if (t >= t_max) break;
    Vector3f pos  = ray(t);
    Float density = sampler.wsSample(openvdb::Vec3d{pos.x(), pos.y(), pos.z()});
    assert(density <= m_maxDensity);
    Float p_s = m_sigma_s * density / max_mu;
    Float p_a = m_sigma_a * density / max_mu;
    Float p_n = 1 - p_s - p_a;
    // if it is sampling some real "balls"
    if (rng.get1D() > p_n) {
      // sample
      vi = VInteraction(pos, -ray.m_d, m_g);
      return Vector3f::Constant(m_sigma_s / m_sigma_t);
    }  // else: continue the rejection process
  }

  return Vector3f::Ones();
}

SV_NAMESPACE_END
