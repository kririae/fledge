#include "integrator.hpp"

SV_NAMESPACE_BEGIN

// call by the class Render
void SampleIntegrator::render(const Scene &scene) {
  Random rng;

  auto resX = scene.m_resX;
  auto resY = scene.m_resY;
  auto SPP  = scene.m_SPP;
  Log("render start with (resX=%d, resY=%d, SPP=%d)", resX, resY, SPP);

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

Vector3f SampleIntegrator::Li(const Ray &ray, const Scene &scene, Random &rng) {
  Vector3f     L = Vector3f::Zero();
  SInteraction isect;
  if (scene.m_accel->intersect(ray, isect))
    L = (isect.m_n + Vector3f::Constant(1.0)) / 2;
  else
    L = Vector3f::Constant(0.00);
  return L;
}

Vector3f PathIntegrator::Li(const Ray &r, const Scene &scene, Random &rng) {
  Vector3f L    = Vector3f::Zero();
  Vector3f beta = Vector3f::Ones();
  auto     ray  = r;
  int      bounces{0};
  bool     specular{false};

  for (bounces = 0;; ++bounces) {
    SInteraction isect;

    bool find_isect = scene.intersect(ray, isect);
    if (bounces == 0 || specular) {
      if (find_isect) {
        L += beta.cwiseProduct(isect.Le());
      } else {
        // environment light
      }
    }

    if (!bounces || bounces >= m_maxDepth) {
      break;
    }
  }

  return L;
}

SV_NAMESPACE_END
