#include "integrator.hpp"

#include <Eigen/src/Core/Reverse.h>

#include "fwd.hpp"
#include "interaction.hpp"
#include "material.hpp"

SV_NAMESPACE_BEGIN

Vector3f EstimateDirect(const Interaction &it, const AreaLight &light,
                        const Vector2f &u_light) {
  Float       pdf;
  Vector3f    wi, L = Vector3f::Zero();
  Interaction light_sample;
  Vector3f    Li = light.sampleLi(it, u_light, wi, pdf, light_sample);
  if (pdf == 0 || Li == Vector3f::Zero()) return Vector3f::Zero();

  CoordinateTransition trans(it.m_ng);
  auto                 t_it = reinterpret_cast<const SInteraction &>(it);
  Vector3f f = t_it.m_primitive->getMaterial()->f(it.m_wo, wi, u_light,
                                                  Vector2f::Zero(), trans);

  if (f != Vector3f::Zero()) {
    auto shadow_ray = t_it.SpawnRayTo(light_sample);
    L += f.cwiseProduct(Li) / pdf;
  }

  return L;
}

Vector3f UniformSampleOneLight(const Interaction &it, const Scene &scene,
                               Random &rng) {
  int n_lights = scene.m_light.size();
  if (n_lights == 0) return Vector3f::Zero();
  int light_num =
      std::min(static_cast<int>(rng.get1D() * static_cast<Float>(n_lights)),
               n_lights - 1);
  Float light_pdf    = 1.0 / n_lights;
  auto  u_light      = rng.get2D();
  auto  u_scattering = rng.get2D();
  return EstimateDirect(it, *(scene.m_light[light_num]), u_light) / light_pdf;
}

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
    L = (isect.m_ns + Vector3f::Constant(1.0)) / 2;
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

  // \sum P(p_n) as a *vector*
  for (bounces = 0;; ++bounces) {
    SInteraction isect;

    bool find_isect = scene.intersect(ray, isect);
    if (bounces == 0 || specular) {
      if (find_isect) {
        L += beta.cwiseProduct(isect.Le(-ray.m_d));
      } else {
        // environment light
      }
    }

    if (!find_isect || bounces >= m_maxDepth) {
      break;
    }

    // consider the *direct lighting*, i.e. L_e terms in LTE
    L += beta.cwiseProduct(UniformSampleOneLight(isect, scene, rng));

    // use the shading normal
    CoordinateTransition trans(isect.m_ns);
    break;
  }

  return L;
}

SV_NAMESPACE_END
