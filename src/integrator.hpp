#ifndef __INTEGRATOR_HPP__
#define __INTEGRATOR_HPP__

#include "fwd.hpp"
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
  void render(const Scene &scene) override {
    Random rng;

    auto resX = scene.m_resX;
    auto resY = scene.m_resY;
    auto SPP  = scene.m_SPP;
    Log("rendering process started with (resX=%d, resY=%d, SPP=%d)", resX, resY,
        SPP);

// TODO: launch worker threads with tbb
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < resX; ++i) {
      for (int j = 0; j < resY; ++j) {
        Vector3f color = Vector3f::Zero();

        // temporary implementation
        for (int s = 0; s < SPP; ++s) {
          auto uv = rng.get2D();
          auto ray =
              scene.m_camera->generateRay(i + uv.x(), j + uv.y(), resX, resY);
          color += Li(ray, scene, rng);
        }

        // store the value back
        scene.m_film->getPixel(i, j) = color / SPP;
      }
    }
  }

  virtual Vector3f Li(const Ray &ray, const Scene &scene, Random &rng) {
    Vector3f     L = Vector3f::Zero();
    SInteraction isect;
    if (scene.m_accel->intersect(ray, isect))
      L = (isect.m_n + Vector3f::Constant(1.0)) / 2;
    else
      L = Vector3f::Constant(0.01);
    return L;
  }
};

class PathIntegrator : public SampleIntegrator {
public:
  PathIntegrator()           = default;
  ~PathIntegrator() override = default;

  void     preprocess() override { TODO(); }
  Vector3f Li(const Ray &ray, const Scene &scene, Random &rng) override {
    return Vector3f::Constant(0.05);
  }
};

SV_NAMESPACE_END

#endif
