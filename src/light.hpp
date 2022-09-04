#ifndef __LIGHT_HPP__
#define __LIGHT_HPP__

#include <future>
#include <memory>

#include "common/vector.h"
#include "debug.hpp"
#include "distribution.hpp"
#include "fledge.h"
#include "interaction.hpp"

FLG_NAMESPACE_BEGIN

class Light {
public:
  virtual ~Light() = default;

  virtual void preprocess(const Scene &scene);
  // given isect on other surface, sample on the light and
  // return the Li arriving at isect
  virtual Vector3f sampleLi(const Interaction &ref, const Vector2f &u,
                            Vector3f &wi, Float &pdf,
                            Interaction &sample) const = 0;
  virtual Float    pdfLi(const Interaction &) const    = 0;

  // interface for infinite light
  virtual Vector3f sampleLe() const;
  virtual Float    pdfLe() const;

  virtual Vector3f Le(const Ray &) const;
};

// diffuse area light
class AreaLight : public Light {
public:
  AreaLight(const std::shared_ptr<Shape> &shape, const Vector3f &Le);
  ~AreaLight() override = default;

  Vector3f sampleLi(const Interaction &ref, const Vector2f &u, Vector3f &wi,
                    Float &pdf, Interaction &sample) const override;
  Float    pdfLi(const Interaction &isect) const override;
  virtual Vector3f L(const Interaction &isect, const Vector3f &w) const;

private:
  std::shared_ptr<Shape> m_shape;
  Vector3f               m_Le;
};

class InfiniteAreaLight : public Light {
public:
  InfiniteAreaLight(const Vector3f &color);
  InfiniteAreaLight(const std::string &filename);
  InfiniteAreaLight(const std::shared_ptr<Texture> &tex);
  ~InfiniteAreaLight() override = default;

  // acquire the limit of the scene
  void     preprocess(const Scene &scene) override;
  Vector3f sampleLi(const Interaction &ref, const Vector2f &u, Vector3f &wi,
                    Float &pdf, Interaction &sample) const override;
  Float    pdfLi(const Interaction &) const override;
  Vector3f Le(const Ray &) const override;

private:
  Vector3f m_worldCenter;
  Float    m_worldRadius;

  std::shared_ptr<Texture> m_tex;
  std::shared_ptr<Dist2D>  m_dist;
  static constexpr int     NU = 512, NV = 512;
};

FLG_NAMESPACE_END

#endif
