#include "volume.hpp"

#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/io/File.h>
#include <openvdb/math/Math.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

#include <memory>

#include "aabb.hpp"
#include "fwd.hpp"
#include "interaction.hpp"

SV_NAMESPACE_BEGIN

VDBVolume::VDBVolume(const std::string &filename) {
  // m_sigma_s and m_sigma_a are decided in advance
  m_sigma_a = 0.0;
  m_sigma_s = 1.0;
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
  auto transform_ptr = m_grid->transformPtr();
  transform_ptr->postRotate(PI, openvdb::math::Y_AXIS);

  // acquire the global maxDensity in nodes
  Float minDensity, maxDensity;
  m_grid->evalMinMax(minDensity, maxDensity);
  SV_Log("m_maxDensity=%f", maxDensity);
  m_maxDensity    = maxDensity * 2;
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
  if (!m_aabb->intersect_pbrt(ray, t_min, t_max)) return Vector3f::Ones();

  // ratio-tracking (I cannot understand it correctness currently)
  Float tr = 1, t = std::max(t_min, static_cast<Float>(0.0));
  t_max        = std::min(t_max, ray.m_tMax);
  Float max_mu = m_sigma_t * m_maxDensity;

  assert(m_aabb->inside(ray(t_max)));

  while (true) {
    t += -std::log(1 - rng.get1D()) / max_mu;
    if (t >= t_max) break;
    Vector3f pos  = ray(t);
    Float density = sampler.wsSample(openvdb::Vec3d{pos.x(), pos.y(), pos.z()});
    tr *= 1 - std::max(static_cast<Float>(0), density * m_invMaxDensity);

    // pbrt's optimization using RR
    const Float rrThreshold = .1;
    if (tr < rrThreshold) {
      Float q = std::max((Float).05, 1 - tr);
      if (rng.get1D() < q) return Vector3f::Zero();
      tr /= 1 - q;
    }
  }

  return Vector3f::Constant(tr);
}

Vector3f VDBVolume::sample(const Ray &ray, Random &rng, VInteraction &vi,
                           bool &success) const {
  // Delta-Tracking
  auto accessor      = m_grid->getConstAccessor();
  using sampler_type = openvdb::tools::Sampler<1>;
  // Extra effort is needed to use openvdb to interpolate
  // for simplicity, we currently will not use openvdb::ray
  auto sampler = openvdb::tools::GridSampler<decltype(accessor), sampler_type>(
      accessor, m_grid->transform());
  Float t_min, t_max;
  if (!m_aabb->intersect_pbrt(ray, t_min, t_max)) {
    success = false;
    return Vector3f::Ones();
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
      return Vector3f::Constant(m_sigma_s / m_sigma_t);
    }  // else: continue the rejection process
  }

  success = false;
  return Vector3f::Ones();
}

// For testing the delta-tracking
HVolume::HVolume() {
  // currently the same as above
  m_sigma_a = 0.0;
  m_sigma_s = 1.0;
  m_g       = -0.3;
  m_sigma_t = m_sigma_a + m_sigma_s;
  m_density = 1.0;

  // m_aabb = std::make_shared<AABB>(Vector3f{-196.66, -68.33, -211.66},
  //                                 Vector3f{218.33, 213.33, 298.33});
  m_aabb = std::make_shared<AABB>(Vector3f{-100, -100, -100},
                                  Vector3f{100, 100, 100});
}

Vector3f HVolume::tr(const Ray &ray, Random &rng) const {
  // calculate the tr from ray.o to ray.m_tMax
  Float t_min, t_max;
  if (!m_aabb->intersect_pbrt(ray, t_min, t_max)) return Vector3f::Ones();
  // clamp the t_max
  t_min = std::max(static_cast<Float>(0), t_min);
  // tr = exp(-t * sigma_t)
  return Vector3f::Constant(std::exp(-(t_max - t_min) * m_density * m_sigma_t));
}

Vector3f HVolume::sample(const Ray &ray, Random &rng, VInteraction &vi,
                         bool &success) const {
  // sample a point inside the volume
  Float t_min, t_max;
  if (!m_aabb->intersect_pbrt(ray, t_min, t_max)) {
    success = false;
    return Vector3f::Ones();
  }

  // sample the distance by $p(t) = \sigma_t e^{-\sigma_t t}$
  Float t = -std::log(1 - rng.get1D()) / (m_density * m_sigma_t);
  // Float tr = std::exp(-t * m_density * m_sigma_t); // tr is elimated
  if (t < t_max) {
    // sampling the volume
    success = true;
    // since we are sampling the volume, and the PDF is exactly p(t)
    vi.m_p  = ray(t);
    vi.m_wo = -ray.m_d;
    vi.m_g  = m_g;
    return Vector3f::Constant(m_sigma_s / m_sigma_t);
  } else {
    // sampling the surface
    success = false;
    // since we are sampling the surface, the result must be divided by the pdf,
    // which is exactly tr
    return Vector3f::Ones();
  }
}

SV_NAMESPACE_END
