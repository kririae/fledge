#ifndef __INTEGRATOR_HPP__
#define __INTEGRATOR_HPP__

#include "fwd.hpp"
#include "ray.hpp"
#include "rng.hpp"

SV_NAMESPACE_BEGIN

class Scene;
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
  int m_maxDepth = 16;
};

// Simple Volume Path Integrator
// only volume in the scene is considered
class SVolIntegrator : public Integrator {
public:
  SVolIntegrator()           = default;
  ~SVolIntegrator() override = default;

  Vector3f Li(const Ray &r, const Scene &scene, Random &rng);

private:
  // maxDepth = 16
};

SV_NAMESPACE_END

#endif
