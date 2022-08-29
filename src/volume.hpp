#ifndef __VOLUME_HPP__
#define __VOLUME_HPP__

#include <openvdb/Grid.h>
#include <openvdb/openvdb.h>

#include <memory>
#include <string>

#include "common/vector.h"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

class Volume {
public:
  virtual ~Volume() = default;

  // the method should return the Transmittance, i.t. T(t) between
  // the ray's origin to ray(tMax). The result might be just estimation
  virtual Vector3f tr(const Ray &ray, Sampler &rng) const = 0;
  // sample a volume interaction inside the volume
  virtual Vector3f sample(const Ray &ray, Sampler &rng, VInteraction &vi,
                          bool &success) const = 0;
  virtual AABB     getBound() const;

  std::shared_ptr<AABB> m_aabb;
};

// homogeneous volume
class HVolume : public Volume {
public:
  HVolume();
  ~HVolume() override = default;

  // function same as above
  Vector3f tr(const Ray &ray, Sampler &rng) const override;
  Vector3f sample(const Ray &ray, Sampler &rng, VInteraction &vi,
                  bool &success) const override;

private:
  Float m_sigma_s, m_sigma_a, m_sigma_t;  // m^2
  Float m_g;
  Float m_density;
};

class OpenVDBVolume : public Volume {
public:
  OpenVDBVolume(const std::string &filename);
  ~OpenVDBVolume() override = default;

  // the method should return the Transmittance, i.t. T(t) between
  // the ray's origin to ray(tMax). The result is just estimation
  Vector3f tr(const Ray &ray, Sampler &rng) const override;
  // sample a volume interaction inside the volume
  Vector3f sample(const Ray &ray, Sampler &rng, VInteraction &vi,
                  bool &success) const override;

private:
  friend class SVolIntegrator;
  Float m_sigma_s, m_sigma_a, m_sigma_t;  // m^2
  // Defined for delta tracking, where mu_{.} = density * sigma_{.}
  Float m_g;
  Float m_maxDensity, m_invMaxDensity;
  // only support one grid currently
  std::shared_ptr<openvdb::FloatGrid> m_grid;
};

class NanoVDBVolume : public Volume {
public:
  NanoVDBVolume(const std::string &filename);
  ~NanoVDBVolume() override = default;

  // the method should return the Transmittance, i.t. T(t) between
  // the ray's origin to ray(tMax). The result is just estimation
  Vector3f tr(const Ray &ray, Sampler &rng) const override;
  // sample a volume interaction inside the volume
  Vector3f sample(const Ray &ray, Sampler &rng, VInteraction &vi,
                  bool &success) const override;

private:
  friend class SVolIntegrator;
  Float m_sigma_s, m_sigma_a, m_sigma_t;  // m^2
  // Defined for delta tracking, where mu_{.} = density * sigma_{.}
  Float m_g;
  Float m_maxDensity, m_invMaxDensity;
  // only support one grid currently
  std::shared_ptr<openvdb::FloatGrid> m_grid;
};

FLG_NAMESPACE_END

#endif
