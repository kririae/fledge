#ifndef __INTEGRATOR_HPP__
#define __INTEGRATOR_HPP__

#include "fwd.hpp"
#include "interaction.hpp"
#include "light.hpp"
#include "ray.hpp"
#include "rng.hpp"
#include "scene.hpp"

SV_NAMESPACE_BEGIN

class Integrator {
public:
  virtual ~Integrator()                   = default;
  virtual void render(const Scene &scene) = 0;
};

// similar code structure to the implementation in pbrt
// [https://pbr-book.org/3ed-2018/Introduction/Class%20Relationships.svg]

class SampleIntegrator : public Integrator {
public:
  SampleIntegrator()           = default;
  ~SampleIntegrator() override = default;

  virtual void preprocess() { TODO(); }
  // call by the class Render
  void             render(const Scene &scene) override;
  virtual Vector3f Li(const Ray &ray, const Scene &scene, Random &rng);
};

class PathIntegrator : public SampleIntegrator {
public:
  PathIntegrator()           = default;
  ~PathIntegrator() override = default;

  void     preprocess() override { TODO(); }
  Vector3f Li(const Ray &r, const Scene &scene, Random &rng) override;

private:
  int m_maxDepth = 32;
};

SV_NAMESPACE_END

#endif
