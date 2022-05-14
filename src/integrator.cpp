#include "integrator.hpp"

#include <Eigen/src/Core/Reverse.h>
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/spin_mutex.h>

#include "camera.hpp"
#include "film.hpp"
#include "fwd.hpp"
#include "interaction.hpp"
#include "light.hpp"
#include "material.hpp"
#include "scene.hpp"
#include "utils.hpp"
#include "volume.hpp"

SV_NAMESPACE_BEGIN

Vector3f EstimateTr(const Ray &ray, const Scene &scene) {
  // only consider the result to be 0 or 1 currently
  SInteraction isect;
  if (scene.intersect(ray, isect)) {
    return Vector3f::Zero();
  } else {
    return Vector3f::Ones();
  }
}

Vector3f VolEstimateTr(const Ray &ray, const Scene &scene, Random &rng) {
  return scene.m_volume->tr(ray, rng);
}

Vector3f EstimateDirect(const Interaction &it, const Light &light,
                        const Scene &scene, Random &rng) {
  Float       pdf;
  Vector3f    wi, L = Vector3f::Zero();
  Vector2f    u_light = rng.get2D();
  Interaction light_sample;
  Vector3f    Li = light.sampleLi(it, u_light, wi, pdf, light_sample);
  if (pdf == 0 || Li == Vector3f::Zero()) return Vector3f::Zero();

  Vector3f f = Vector3f::Zero();
  if (it.isSInteraction()) {
    // Surface Interaction
    CoordinateTransition trans(it.m_ng);
    auto                 t_it = reinterpret_cast<const SInteraction &>(it);
    assert(t_it.m_primitive->getMaterial() != nullptr);
    f = t_it.m_primitive->getMaterial()->f(it.m_wo, wi, Vector2f::Zero(),
                                           trans) *
        abs(wi.dot(t_it.m_ns));
  } else {
    // Volume Interaction
    VInteraction vit = reinterpret_cast<const VInteraction &>(it);
    // likely the BRDF to be applied
    f = Vector3f::Constant(HGP(wi, vit.m_wo, vit.m_g));
  }

  if (f != Vector3f::Zero()) {
    // Notice that *SpawnRayTo* is responsible for initializing the
    // ray.tMax, so if intersection
    auto     shadow_ray = it.SpawnRayTo(light_sample);
    Vector3f tr         = VolEstimateTr(shadow_ray, scene, rng);
    Li                  = Li.cwiseProduct(tr);
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
  Float light_pdf = 1.0 / n_lights;
  return EstimateDirect(it, *(scene.m_light[light_num]), scene, rng) /
         light_pdf;
}

// call by the class Render
void SampleIntegrator::render(const Scene &scene) {
  Random rng;

  auto resX = scene.m_resX;
  auto resY = scene.m_resY;
  auto SPP  = scene.m_SPP;
  SV_Log("render start with (resX=%d, resY=%d, SPP=%d)", resX, resY, SPP);

  constexpr int BLOCK_SIZE = 16;
#ifdef USE_TBB
  SV_Log("TBB is ready");
  int blockX = (resX - 1) / BLOCK_SIZE + 1;
  int blockY = (resY - 1) / BLOCK_SIZE + 1;
  // clang-format off
  int cnt = 0;

  oneapi::tbb::spin_mutex l_mutex;

  // currently using tbb might lead to a performance degradation
  tbb::parallel_for(tbb::blocked_range2d<int, int>(0, blockX, 0, blockY),
    [resX, resY, SPP, blockX, blockY, &scene, &rng, &cnt, &l_mutex,
      this](const tbb::blocked_range2d<int, int> &r) {
      int r_l = r.rows().begin() * BLOCK_SIZE,
          r_r = std::min(r.rows().end() * BLOCK_SIZE, resX);
      int c_l = r.cols().begin() * BLOCK_SIZE,
          c_r = std::min(r.cols().end() * BLOCK_SIZE, resY);
      for (auto i = r_l; i < r_r; ++i) {
        for (auto j = c_l; j < c_r; ++j) {
          Vector3f color = Vector3f::Zero();

          // temporary implementation
          for (int s = 0; s < SPP; ++s) {
            auto uv  = rng.get2D();
            auto ray = scene.m_camera->generateRay(
                i + uv.x(), j + uv.y(), resX, resY);
            color += Li(ray, scene, rng);
          }

          // store the value back
          scene.m_film->getPixel(i, j) = color / SPP;
        }
      }

      { // scope
        tbb::spin_mutex::scoped_lock lock(l_mutex);
        ++cnt;
        // potential bug
        SV_Log("[%d/%d] blocks are finished", cnt, blockX * blockY);
        if (cnt % 50 == 0) {
          scene.m_film->saveImage("tmp.exr");
        }
      }
    });
  // clang-format on
#else
  std::atomic<int> cnt;
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
#pragma omp critical  // for output
      {
        ++cnt;
        printf("rendering: %f\r", static_cast<Float>(cnt) / resX / resY);
        if (i % 100 == 0 && j == 0) {
          scene.m_film->saveImage("tmp.exr");
        }
      }
    }
  }
#endif
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
        for (const auto &light : scene.m_infLight) {
          L += beta.cwiseProduct(light->Le(ray));
        }
      }
    }

    if (!find_isect || bounces >= m_maxDepth) {
      break;
    }

    // consider the *direct lighting*, i.e. L_e terms in LTE
    L += beta.cwiseProduct(UniformSampleOneLight(isect, scene, rng));

    // use the shading normal
    CoordinateTransition trans(isect.m_ns);

    // spawn ray to new direction
    Vector3f wi, wo = -ray.m_d;
    Float    pdf;
    Vector3f f = isect.m_primitive->getMaterial()->sampleF(
        wo, wi, pdf, rng.get2D(), Vector2f::Zero(), trans);
    if (pdf == 0.0 || f.isZero()) break;

    beta = beta.cwiseProduct(f * abs(wi.dot(isect.m_ns)) / pdf);
    ray  = isect.SpawnRay(wi);
  }

  return L;
}

Vector3f SVolIntegrator::Li(const Ray &r, const Scene &scene, Random &rng) {
  Vector3f L    = Vector3f::Zero();
  Vector3f beta = Vector3f::Ones();
  auto     ray  = r;
  bool     in_volume{false};
  int      bounces{0};

  // only use the "scene.m_volume" for integrating
  for (bounces = 0;; bounces++) {
    VInteraction vit;

    Float t_min, t_max;
    bool  find_isect = scene.m_volume->m_aabb->intersect(ray, t_min, t_max);
    if (bounces == 0) {
      if (find_isect) {
        // consider T(p, p_e) L_o(p_e, -w), i.e. P(p_0)
        auto tr = scene.m_volume->tr(ray, rng);
        for (const auto &light : scene.m_infLight)
          L += tr.cwiseProduct(light->Le(ray));
        in_volume = true;

        assert(t_min >= 0);
        auto pos = ray(t_min);
        // SpawnRay manually
        ray = Ray(pos, ray.m_d);
        continue;
      } else {
        // environment light
        for (const auto &light : scene.m_infLight)
          L += beta.cwiseProduct(light->Le(ray));
        break;
      }
    }

    if (in_volume) {
      bool success{false};
      auto f = scene.m_volume->sample(ray, rng, vit, success);
      if (success) {
        // if it is in volume(so t_min <= 0), and the sample is in the volume
        // T \mu s are take into consider by the Estimator
        beta = beta.cwiseProduct(f);

        Vector3f wi;
        HGSampleP(vit.m_wo, wi, rng.get1D(), rng.get1D(), vit.m_g);
        L += beta.cwiseProduct(UniformSampleOneLight(vit, scene, rng));

        if (bounces >= m_maxDepth) break;
        ray = vit.SpawnRay(wi);
      } else {
        // The sample is not permitted since it is already considered in the
        // UniformSampleOneLight(and bounces == 1)
        break;
      }
    }

    if (bounces > 3) {
      if (rng.get1D() < m_rrThreshold) {
        break;
      }

      beta /= (1 - m_rrThreshold);
    }
  }

  return L;
}

SV_NAMESPACE_END
