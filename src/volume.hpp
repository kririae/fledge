#ifndef __VOLUME_HPP__
#define __VOLUME_HPP__

#include <openvdb/Grid.h>
#include <openvdb/openvdb.h>

#include <memory>
#include <string>

#include "aabb.hpp"
#include "fwd.hpp"
#include "integrator.hpp"
#include "interaction.hpp"
#include "rng.hpp"

SV_NAMESPACE_BEGIN

class Volume {
public:
  virtual ~Volume() = default;

  // the method should return the Transmittance, i.t. T(t) between
  // the ray's origin to ray(tMax). The result might be just estimation
  virtual Vector3f tr(const Ray &ray, Random &rng) const = 0;
  // sample a volume interaction inside the volume
  virtual Vector3f sample(const Ray &ray, Random &rng,
                          VInteraction &vi) const = 0;
};

class VDBVolume : public Volume {
public:
  VDBVolume(const std::string &filename);
  ~VDBVolume() override = default;

  // the method should return the Transmittance, i.t. T(t) between
  // the ray's origin to ray(tMax). The result is just estimation
  Vector3f tr(const Ray &ray, Random &rng) const override;
  // sample a volume interaction inside the volume
  Vector3f sample(const Ray &ray, Random &rng, VInteraction &vi) const override;

private:
  friend class SVolIntegrator;
  Float m_sigma_s, m_sigma_a, m_sigma_t;  // m^2
  // Defined for delta tracking, where mu_{.} = density * sigma_{.}
  Float                 m_mu_s, m_mu_a, m_mu_t;
  Float                 m_g;
  Float                 m_maxDensity, m_invMaxDensity;
  std::shared_ptr<AABB> m_aabb;
  // only support one grid currently
  std::shared_ptr<openvdb::FloatGrid> m_grid;
};

SV_NAMESPACE_END

#endif
