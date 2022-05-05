#ifndef __LIGHT_HPP__
#define __LIGHT_HPP__

#include <future>
#include <memory>

#include "fwd.hpp"
#include "interaction.hpp"
#include "shape.hpp"

SV_NAMESPACE_BEGIN

class Light {
public:
  virtual ~Light() = default;
  // given isect on other surface, sample on the light and
  // return the Li arriving at isect
  virtual Vector3f sampleLi(const Interaction &ref, const Vector2f &u,
                            Vector3f &wi, Float &pdf) = 0;
  virtual Float    pdfLi(const Interaction &)         = 0;

  // interface for infinite light
  virtual Vector3f sampleLe() const;
  virtual Float    pdfLe() const;
};

// diffuse area light
class AreaLight : public Light {
public:
  AreaLight(const std::shared_ptr<Shape> &shape, const Vector3f &Le);
  ~AreaLight() override = default;

  Vector3f sampleLi(const Interaction &ref, const Vector2f &u, Vector3f &wi,
                    Float &pdf) override;
  Float    pdfLi(const Interaction &isect) override;
  virtual Vector3f L(const Interaction &isect) const;

private:
  std::shared_ptr<Shape> m_shape;
  Vector3f               m_Le;
};

SV_NAMESPACE_END

#endif
