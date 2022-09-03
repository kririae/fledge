#include "volume.hpp"

#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/io/File.h>
#include <openvdb/math/Math.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

#include <memory>

#include "common/aabb.h"
#include "common/ray.h"
#include "common/sampler.h"
#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"
#include "interaction.hpp"
#include "rng.hpp"

FLG_NAMESPACE_BEGIN

AABB Volume::getBound() const {
  return m_aabb;
}

// This method is intended to be invoked by primitive
// mostly for Homogeneous Volume
void Volume::setBound(const AABB &aabb) {
  m_aabb = aabb;
}

OpenVDBVolume::OpenVDBVolume(const std::string &filename) {
  // m_sigma_s and m_sigma_a are decided in advance
  m_sigma_s = 10.0;
  m_sigma_a = 1.0;
  m_g       = 0.877;
  m_sigma_t = m_sigma_a + m_sigma_s;

  openvdb::initialize();
  openvdb::io::File file(filename);
  SLog("load OpenVDB file %s", filename.c_str());
  file.open();
  // the only grid we need from cloud volume
  auto baseGrid = file.readGrid("density");
  m_grid        = std::shared_ptr<openvdb::FloatGrid>(
      openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid));
  auto transform_ptr = m_grid->transformPtr();
  transform_ptr->postRotate(PI, openvdb::math::Y_AXIS);

  // acquire the global maxDensity in nodes
  Float minDensity, maxDensity;
  // m_grid->evalMinMax(minDensity, maxDensity);
  auto minMax = openvdb::tools::minMax(m_grid->tree());
  minDensity  = minMax.min();
  maxDensity  = minMax.max();
  SLog("m_minDensity=%f, m_maxDensity=%f", minDensity, maxDensity);
  m_maxDensity    = maxDensity;
  m_invMaxDensity = 1.0 / m_maxDensity;

  auto aabb  = m_grid->evalActiveVoxelBoundingBox();
  auto l_min = aabb.min(), l_max = aabb.max();
  auto w_min = m_grid->indexToWorld(l_min);
  auto w_max = m_grid->indexToWorld(l_max);
  SLog("AABB low=(%f,%f,%f) max=(%f,%f,%f)", w_min.x(), w_min.y(), w_min.z(),
       w_max.x(), w_max.y(), w_max.z());

  // initialize here
  m_aabb = AABB(Vector3f{Float(w_min[0]), Float(w_min[1]), Float(w_min[2])},
                Vector3f{Float(w_max[0]), Float(w_max[1]), Float(w_max[2])});

  file.close();
}

Vector3f OpenVDBVolume::tr(const Ray &ray, Sampler &rng) const {
  auto accessor      = m_grid->getConstAccessor();
  using sampler_type = openvdb::tools::Sampler<1>;
  // Extra effort is needed to use openvdb to interpolate
  // for simplicity, we currently will not use openvdb::ray
  auto sampler = openvdb::tools::GridSampler<decltype(accessor), sampler_type>(
      accessor, m_grid->transform());
  Float t_min, t_max;
  if (!m_aabb.intersect_pbrt(ray, t_min, t_max)) return Vector3f(1.0);

  // ratio-tracking (I cannot understand it correctness currently)
  Float tr = 1, t = std::max(t_min, static_cast<Float>(0.0));
  t_max        = std::min(t_max, ray.m_tMax);
  Float max_mu = m_sigma_t * m_maxDensity;

  while (true) {
    t += -std::log(1 - rng.get1D()) / max_mu;
    if (t >= t_max) break;
    Vector3f pos  = ray(t);
    Float density = sampler.wsSample(openvdb::Vec3d{pos.x(), pos.y(), pos.z()});
    tr *= 1 - std::max(static_cast<Float>(0), density * m_invMaxDensity);

    // PBRT's optimization using RR
    const Float rrThreshold = .1;
    if (tr < rrThreshold) {
      Float q = std::max((Float).05, 1 - tr);
      if (rng.get1D() < q) return Vector3f(0.0);
      tr /= 1 - q;
    }
  }

  return Vector3f(tr);
}

Vector3f OpenVDBVolume::sample(const Ray &ray, Sampler &rng, VInteraction &vi,
                               bool &success) const {
  // Delta-Tracking
  auto accessor      = m_grid->getConstAccessor();
  using sampler_type = openvdb::tools::Sampler<1>;
  // Extra effort is needed to use openvdb to interpolate
  // for simplicity, we currently will not use openvdb::ray
  auto sampler = openvdb::tools::GridSampler<decltype(accessor), sampler_type>(
      accessor, m_grid->transform());
  Float t_min, t_max;
  if (!m_aabb.intersect_pbrt(ray, t_min, t_max)) {
    success = false;
    return Vector3f(1.0);
  }

  Float t      = std::max(t_min, static_cast<Float>(0));
  Float max_mu = m_sigma_t * m_maxDensity;
  t_max        = std::min(t_max, ray.m_tMax);
  while (true) {
    // free-path sampling
    // delta-tracking in \overline{\mu}
    t += -std::log(1 - rng.get1D()) / max_mu;
    if (t >= t_max) break;
    Vector3f pos  = ray(t);
    Float density = sampler.wsSample(openvdb::Vec3d{pos.x(), pos.y(), pos.z()});
    Float p_s     = m_sigma_s * density / max_mu;
    Float p_a     = m_sigma_a * density / max_mu;
    Float p_n     = 1 - p_s - p_a;
    // if it is sampling some real "balls"
    if (rng.get1D() > p_n) {
      // sample
      success = true;
      vi      = VInteraction(pos, -ray.m_d, m_g);
      return Vector3f(m_sigma_s / m_sigma_t);
    }  // else: continue the rejection process
  }

  success = false;
  return Vector3f(1.0);
}

// For testing the delta-tracking
HVolume::HVolume(const Vector3f &sigma_s, const Vector3f &sigma_a, Float g,
                 Float density)
    : m_sigma_s(sigma_s),
      m_sigma_a(sigma_a),
      m_sigma_t(sigma_s + sigma_a),
      m_g(g),
      m_density(density) {}

Vector3f HVolume::tr(const Ray &ray, Sampler &sampler) const {
  return Exp(-ray.m_tMax * m_density * m_sigma_t);
}

// @INIT_INTERACTION
Vector3f HVolume::sample(const Ray &ray, Sampler &sampler, VInteraction &vi,
                         bool &success) const {
  // First, sample a channel
  int channel = min((int)(sampler.get1D() * 3.0), 2);
  // sample the distance by $p(t) = \sigma_t e^{-\sigma_t t}$
  Float t = -std::log(1 - sampler.get1D()) / (m_density * m_sigma_t[channel]);
  Vector3f Tr = Exp(-min(t, ray.m_tMax) * m_density *
                    m_sigma_t);  // using t rather than t_Max
  C(t);
  if (t < ray.m_tMax) {
    Float pdf = Sum(m_sigma_t * Tr) * 1 / 3;
    // sampling the volume
    success = true;
    // since we are sampling the volume, and the PDF is exactly p(t)
    vi.m_ray = ray;
    vi.m_p   = ray(t);
    vi.m_wo  = -ray.m_d;
    vi.m_g   = m_g;
    return m_sigma_s * Tr / pdf;
  } else {
    Float pdf = Sum(Tr) * 1 / 3;
    // sampling the surface
    success = false;
    return Tr / pdf;
  }
}

FLG_NAMESPACE_END
