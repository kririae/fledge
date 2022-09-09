#ifndef __INTEGRATOR_HPP__
#define __INTEGRATOR_HPP__

#include "debug.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

class Scene;
class Integrator {
public:
  virtual ~Integrator()                   = default;
  virtual void render(const Scene &scene) = 0;
};

// similar code structure to the implementation in pbrt
// [https://pbr-book.org/3ed-2018/Introduction/Class%20Relationships.svg]
class ParallelIntegrator : public Integrator {
public:
  ParallelIntegrator()           = default;
  ~ParallelIntegrator() override = default;

  virtual void preprocess() { TODO(); }
  // call by the class Render
  void             render(const Scene &scene) override;
  virtual Vector3f Li(const Ray &ray, const Scene &scene, Sampler &sampler,
                      Vector3f *albedo = nullptr, Vector3f *normal = nullptr);

private:
};

class PathIntegrator : public ParallelIntegrator {
public:
  PathIntegrator(int max_depth) : m_maxDepth(max_depth) {}
  ~PathIntegrator() override = default;

  void     preprocess() override { TODO(); }
  Vector3f Li(const Ray &r, const Scene &scene, Sampler &Sampler,
              Vector3f *albedo = nullptr, Vector3f *normal = nullptr) override;

private:
  int m_maxDepth = 16;
};

// Simple Volume Path Integrator
// only volume in the scene is considered
class SVolIntegrator : public ParallelIntegrator {
public:
  SVolIntegrator(int max_depth) : m_maxDepth(max_depth) {}
  ~SVolIntegrator() override = default;

  Vector3f Li(const Ray &r, const Scene &scene, Sampler &Sampler,
              Vector3f *albedo = nullptr, Vector3f *normal = nullptr) override;

private:
  int   m_maxDepth    = 512;
  Float m_rrThreshold = 0.01;
};

class VolPathIntegrator : public ParallelIntegrator {
public:
  VolPathIntegrator(int max_depth) : m_maxDepth(max_depth) {}
  ~VolPathIntegrator() override = default;

  Vector3f Li(const Ray &r, const Scene &scene, Sampler &sampler,
              Vector3f *albedo = nullptr, Vector3f *normal = nullptr) override;

private:
  int m_maxDepth = 512;
};

FLG_NAMESPACE_END

#endif
