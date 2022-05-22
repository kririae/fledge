#include "integrator.hpp"

#include <Eigen/src/Core/Reverse.h>

#include <chrono>
#include <cstddef>
#include <thread>

#include "aabb.hpp"
#include "camera.hpp"
#include "debug.hpp"
#include "film.hpp"
#include "fwd.hpp"
#include "interaction.hpp"
#include "light.hpp"
#include "material.hpp"
#include "primitive.hpp"
#include "rng.hpp"
#include "scene.hpp"
#include "utils.hpp"
#include "volume.hpp"

SV_NAMESPACE_BEGIN

Vector3f EstimateTr(const Ray &ray, const Scene &scene, Random &rng) {
  // only consider the result to be 0 or 1 currently
  SInteraction isect;
  if (scene.intersect(ray, isect)) {
    return Vector3f::Zero();
  } else {
    return Vector3f::Ones();
  }
}

Vector3f VolEstimateTr(const Ray &ray, const Scene &scene, Random &rng) {
  CPtr(scene.m_volume);
  CNorm(ray.m_d);
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
    // SErr("do not support SInteraction yet");
  } else {
    // Volume Interaction
    VInteraction vit = reinterpret_cast<const VInteraction &>(it);

    CFloat(vit.m_g);
    CNorm(vit.m_wo);
    CNorm(wi);
    // likely the BRDF to be applied
    f = Vector3f::Constant(HGP(wi, vit.m_wo, vit.m_g));
    CVec3(f);
  }

  if (f != Vector3f::Zero()) {
    // Notice that *SpawnRayTo* is responsible for initializing the
    // ray.tMax, so if intersection
    auto     shadow_ray = it.SpawnRayTo(light_sample);
    Vector3f tr         = VolEstimateTr(shadow_ray, scene, rng);
    // CVec3(tr);
    Li = Li.cwiseProduct(tr);
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
  constexpr int BLOCK_SIZE    = 16;
  constexpr int SAVE_INTERVAL = 20;  // (s)
  auto          resX          = scene.m_resX;
  auto          resY          = scene.m_resY;
  auto          SPP           = scene.m_SPP;
  auto          blockX        = (resX - 1) / BLOCK_SIZE + 1;
  auto          blockY        = (resY - 1) / BLOCK_SIZE + 1;
  SLog("render start with (resX=%d, resY=%d, SPP=%d)", resX, resY, SPP);
  SLog("parallel info: (BLOCK_SIZE=%d, blockX=%d, blockY=%d)", BLOCK_SIZE,
       blockX, blockY);

  // define to lambdas here for further evaluation
  auto evalPixel = [&](int x, int y, int SPP, Random &rng) -> Vector3f {
    Vector3f color = Vector3f::Zero();

    // temporary implementation
    for (int s = 0; s < SPP; ++s) {
      auto uv = rng.get2D();
      auto ray =
          scene.m_camera->generateRay(x + uv.x(), y + uv.y(), resX, resY);
      color += Li(ray, scene, rng);
    }

    return color / SPP;
  };

  // to be paralleled
  // starting from (x, y)
  auto evalBlock = [&](int x, int y, int width, int height, int SPP) {
    Random rng(y * resX + x);
    int    x_max = std::min(x + width, scene.m_resX);
    int    y_max = std::min(y + height, scene.m_resY);
    for (int i = x; i < x_max; ++i) {
      for (int j = y; j < y_max; ++j) {
        scene.m_film->getPixel(i, j) = evalPixel(i, j, SPP, rng);
      }
    }
  };

  int  block_cnt = 0;
  auto start     = std::chrono::system_clock::now();

#ifdef USE_TBB
  static_assert(false, "TBB is not supported yet");
  SLog("TBB is ready");
  auto __tbb_evalBlock = [&](const tbb::blocked_range2d<int, int> &r) {
    // r specifies a range in blocks
    for (int i = r.rows().begin(); i < r.rows().end(); ++i) {
      for (int j = r.cols().begin(); j < r.cols().end(); ++j) {
        int x = i * BLOCK_SIZE, y = j * BLOCK_SIZE;
        evalBlock(x, y, BLOCK_SIZE, BLOCK_SIZE, SPP);
      }
    }
  };
  tbb::parallel_for(tbb::blocked_range2d<int, int>(0, blockX, 0, blockY),
                    __tbb_evalBlock);
#else
  SLog("OpenMP is ready");
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
  for (int i = 0; i < blockX; ++i) {
    for (int j = 0; j < blockY; ++j) {
      int x = i * BLOCK_SIZE, y = j * BLOCK_SIZE;
      evalBlock(x, y, BLOCK_SIZE, BLOCK_SIZE, SPP);

#pragma omp critical
      {
        ++block_cnt;
        if (block_cnt % 100 == 0)
          SLog("[%d/%d] blocks are finished", block_cnt, blockX * blockY);
        // save when time passed
        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = end - start;
        if (elapsed_seconds.count() > SAVE_INTERVAL) {
          scene.m_film->saveImage("smallvol_out.exr");
          start = std::chrono::system_clock::now();
        }
      }  // omp critical
    }
  }  // omp loop
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
  ARayCount("Li");

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

      CPtr(scene.m_volume);
      auto f = scene.m_volume->sample(ray, rng, vit, success);
      CVec3(f);

      if (success) {
        // if it is in volume(so t_min <= 0), and the sample is in the volume
        // T \mu s are take into consider by the Estimator
        beta = beta.cwiseProduct(f);
        CVec3(beta);

        Vector3f wi;
        HGSampleP(vit.m_wo, wi, rng.get1D(), rng.get1D(), vit.m_g);

        L += beta.cwiseProduct(UniformSampleOneLight(vit, scene, rng));
        CVec3(L);

        if (bounces >= m_maxDepth || beta.isZero()) break;
        ray = vit.SpawnRay(wi);
      } else {
        // The sample is not permitted since it is already considered in the
        // UniformSampleOneLight(and bounces == 1)
        break;
      }
    }

    // if (bounces > 3) {
    //   if (rng.get1D() < m_rrThreshold) {
    //     break;
    //   }

    //   beta /= (1 - m_rrThreshold);
    // }
  }

  return L;
}

SV_NAMESPACE_END
