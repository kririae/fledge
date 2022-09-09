#include "integrator.hpp"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <thread>

#include "common/aabb.h"
#include "common/camera.h"
#include "common/math_utils.h"
#include "common/sampler.h"
#include "common/vector.h"
#include "debug.hpp"
#include "film.hpp"
#include "fledge.h"
#include "interaction.hpp"
#include "light.hpp"
#include "materials/material.hpp"
#include "primitive.hpp"
#include "rng.hpp"
#include "scene.hpp"
#include "volume.hpp"

FLG_NAMESPACE_BEGIN

Vector3f EstimateTr(const Ray &ray, const Scene &scene, Sampler &sampler) {
  // only consider the result to be 0 or 1 currently
  SInteraction isect;
  if (scene.intersect(ray, isect)) {
    return Vector3f(0.0);
  } else {
    return Vector3f(1.0);
  }
}

/**
 * Starting from ray's origin, estimate Tr until ray(m_tMax) */
Vector3f VolEstimateTr(Ray ray, const Scene &scene, Sampler &sampler) {
  Float    t_max = ray.m_tMax;
  Vector3f tr(1.0);

  while (true) {
    SInteraction isect;
    // recursively evaluate tr
    bool find_isect = scene.intersect(ray, isect);
    if (find_isect) {
      assert(isect.isSInteraction());
      // The behavior of shading the intersection is undefined
      // (and it can not have any volume)
      if (isect.m_primitive == nullptr) return 0.0;
      // If it has some material, then it should hinder the light transport
      if (isect.m_primitive->getMaterial() != nullptr) return 0.0;
      if (ray.m_volume != nullptr) tr *= ray.m_volume->tr(ray, sampler);
      t_max -= ray.m_tMax;
      if (t_max < 0) return tr;
      ray        = isect.SpawnRay(ray.m_d);
      ray.m_tMax = t_max;
    } else {
      return tr;
    }
  }

  return tr;
}

Vector3f EstimateDirect(const Interaction &it, const Light &light,
                        const Scene &scene, Sampler &sampler) {
  Float       pdf;
  Vector3f    wi, L = Vector3f(0.0);
  Vector2f    u_light = sampler.get2D();
  Interaction light_sample;
  Vector3f    Li = light.sampleLi(it, u_light, wi, pdf, light_sample);
  if (pdf == 0 || Li == Vector3f(0.0)) return Vector3f(0.0);

  Vector3f f = Vector3f(0.0);
  if (it.isSInteraction()) {
    // Surface Interaction
    CoordinateTransition trans(it.m_ng);
    auto                 t_it = reinterpret_cast<const SInteraction &>(it);
    assert(t_it.m_primitive->getMaterial() != nullptr);
    f = t_it.m_primitive->getMaterial()->f(it.m_wo, wi, Vector2f(0.0), trans) *
        abs(Dot(wi, t_it.m_ns));
  } else {
    // Volume Interaction
    VInteraction vit = reinterpret_cast<const VInteraction &>(it);

    C(vit.m_g, vit.m_wo, wi);
    // likely the BRDF to be applied
    f = Vector3f(HGP(wi, vit.m_wo, vit.m_g));
    C(f);
  }

  if (f != Vector3f(0.0)) {
    // Notice that *SpawnRayTo* is responsible for initializing the
    // ray.tMax, so if intersection
    auto     shadow_ray = it.SpawnRayTo(light_sample);
    Vector3f tr         = VolEstimateTr(shadow_ray, scene, sampler);
    C(tr);
    Li *= tr;
    L += f * Li / pdf;
  }

  return L;
}

Vector3f UniformSampleOneLight(const Interaction &it, const Scene &scene,
                               Sampler &sampler) {
  int n_lights = scene.m_light.size();
  if (n_lights == 0) return Vector3f(0.0);
  int light_num =
      std::min(static_cast<int>(sampler.get1D() * static_cast<Float>(n_lights)),
               n_lights - 1);
  Float light_pdf = 1.0 / n_lights;
  assert(light_pdf == 1.0);
  return EstimateDirect(it, *(scene.m_light[light_num]), scene, sampler) /
         light_pdf;
}

// call by the class Render
void ParallelIntegrator::render(const Scene &scene) {
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
  auto evalPixel = [&](int x, int y, int SPP, Vector3f *albedo = nullptr,
                       Vector3f *normal = nullptr) -> Vector3f {
    Sampler sampler(SPP, x + y * resX);
    sampler.setPixel(Vector2f(x + 0.5, y + 0.5));
    Vector3f color = Vector3f(0.0);
    sampler.reset();  // start generating

    auto ray = scene.m_camera->generateRay(x + 0.5, y + 0.5, resX, resY);
    color += Li(ray, scene, sampler, albedo, normal);
    // temporary implementation
    for (int s = 1; s < SPP; ++s) {
      auto uv  = sampler.getPixelSample();
      auto ray = scene.m_camera->generateRay(uv.x(), uv.y(), resX, resY);
      color += Li(ray, scene, sampler);
      sampler.reset();
    }

    return color / SPP;
  };

  // to be paralleled
  // starting from (x, y)
  auto evalBlock = [&](int x, int y, int width, int height, int SPP) {
    int x_max = std::min(x + width, scene.m_resX);
    int y_max = std::min(y + height, scene.m_resY);
    for (int i = x; i < x_max; ++i) {
      for (int j = y; j < y_max; ++j) {
        Vector3f albedo, normal;
        scene.m_film->getBuffer(i, j, EFilmBufferType::EColor) =
            evalPixel(i, j, SPP, &albedo, &normal);
        scene.m_film->getBuffer(i, j, EFilmBufferType::EAlbedo) = albedo;
        scene.m_film->getBuffer(i, j, EFilmBufferType::ENormal) = normal;
      }
    }
  };

  int  block_cnt = 0;
  auto start     = std::chrono::system_clock::now();

#ifdef FLEDGE_USE_TBB
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
          scene.m_film->saveBuffer("tmp.exr", EFilmBufferType::EColor);
          start = std::chrono::system_clock::now();
        }
      }  // omp critical
    }
  }  // omp loop
#endif
}

Vector3f ParallelIntegrator::Li(const Ray &ray, const Scene &scene,
                                Sampler &sampler, Vector3f *albedo,
                                Vector3f *normal) {
  Vector3f     L = Vector3f(0.0);
  SInteraction isect;
  if (albedo != nullptr) *albedo = Vector3f(0.0);
  if (normal != nullptr) *normal = Vector3f(0.0);
  if (scene.m_accel->intersect(ray, isect))
    L = (isect.m_ns + Vector3f(1.0)) / 2;
  else
    L = Vector3f(0.00);
  return L;
}

Vector3f PathIntegrator::Li(const Ray &r, const Scene &scene, Sampler &sampler,
                            Vector3f *albedo, Vector3f *normal) {
  Vector3f L    = Vector3f(0.0);
  Vector3f beta = Vector3f(1.0);
  auto     ray  = r;
  int      bounces{0};
  bool     specular{false};

  Vector3f p    = r.m_o;
  Float    rate = 1.0;
  // \sum P(p_n) as a *vector*
  for (bounces = 0;; ++bounces) {
    SInteraction isect;

    bool find_isect = scene.intersect(ray, isect);
    if (find_isect) {
      rate += (isect.m_p - p).norm();
      p = isect.m_p;
    }

    // Handle albedo and normal
    if (bounces == 0) {
      if (find_isect) {
        if (normal != nullptr) *normal = isect.m_ns;
        if (albedo != nullptr)
          *albedo = isect.m_primitive->getMaterial()->getAlbedo();
      } else {
        // no intersection
        if (normal != nullptr) *normal = -ray.m_d;
        if (albedo != nullptr) {
          *albedo = Vector3f(0.0);
          for (const auto &light : scene.m_infLight)
            *albedo += beta * light->Le(ray);
        }
      }
    }

    if (bounces == 0 || specular) {
      if (find_isect) {
        L += beta * isect.Le(-ray.m_d);
      } else {
        // environment light
        for (const auto &light : scene.m_infLight) L += beta * light->Le(ray);
      }
    }

    if (!find_isect || bounces >= m_maxDepth) {
      break;
    }

    // Handling intersection with specular material
    if (!isect.m_primitive->getMaterial()->isDelta()) {
      // consider the *direct lighting*, i.e. L_e terms in LTE
      L += beta * UniformSampleOneLight(isect, scene, sampler);
    } else {
      specular = true;
    }

    // use the shading normal
    CoordinateTransition trans(isect.m_ns);

    // spawn ray to new direction
    Vector3f wi, wo = -ray.m_d;
    Float    pdf;
    C(isect.m_primitive);
    Vector3f f = isect.m_primitive->getMaterial()->sampleF(
        wo, wi, pdf, sampler.get2D(), Vector2f(0.0), trans);
    if (pdf == 0.0 || f.isZero()) break;

    beta = beta * f * abs(Dot(wi, isect.m_ns)) / pdf;
    ray  = isect.SpawnRay(wi);
  }

  return L;
}

Vector3f SVolIntegrator::Li(const Ray &r, const Scene &scene, Sampler &sampler,
                            Vector3f *albedo, Vector3f *normal) {
  Vector3f L    = Vector3f(0.0);
  Vector3f beta = Vector3f(1.0);
  auto     ray  = r;
  bool     in_volume{false};
  int      bounces{0};

  if (albedo != nullptr) *albedo = Vector3f(0.0);
  if (normal != nullptr) *normal = Vector3f(0.0);

  // only use the "scene.m_volume" for integrating
  for (bounces = 0;; bounces++) {
    VInteraction vit;

    Float t_min, t_max;
    bool  find_isect = scene.m_volume->m_aabb.intersect(ray, t_min, t_max);
    if (bounces == 0) {
      if (find_isect) {
        // consider T(p, p_e) L_o(p_e, -w), i.e. P(p_0)
        auto tr = scene.m_volume->tr(ray, sampler);
        for (const auto &light : scene.m_infLight) L += tr * light->Le(ray);
        in_volume = true;

        assert(t_min >= 0);
        auto pos = ray(t_min);
        // SpawnRay manually
        ray = Ray(pos, ray.m_d);
        continue;
      } else {
        // environment light
        for (const auto &light : scene.m_infLight) L += beta * light->Le(ray);
        goto sample_environment;
      }
    }

    if (in_volume) {
      bool success{false};

      C(scene.m_volume);
      auto f = scene.m_volume->sample(ray, sampler, vit, success);
      C(f);

      if (success) {
        // if it is in volume(so t_min <= 0), and the sample is in the volume
        // T \mu s are take into consider by the Estimator
        beta = beta * f;
        C(beta);

        Vector3f wi;
        HGSampleP(vit.m_wo, wi, sampler.get1D(), sampler.get1D(), vit.m_g);

        L += beta * UniformSampleOneLight(vit, scene, sampler);
        C(L);

        if (bounces >= m_maxDepth || beta.isZero()) break;
        ray = vit.SpawnRay(wi);
      } else {
        // The sample is not permitted since it is already considered in the
        // UniformSampleOneLight(and bounces == 1)
        goto sample_surface;
      }
    }

    if (bounces > 10) {
      if (sampler.get1D() < m_rrThreshold) {
        break;
      }

      beta /= (1 - m_rrThreshold);
    }
  }

sample_surface:
sample_environment:  // do not count this case

  return L;
}

Vector3f VolPathIntegrator::Li(const Ray &r, const Scene &scene,
                               Sampler &sampler, Vector3f *albedo,
                               Vector3f *normal) {
  Vector3f L    = Vector3f(0.0);
  Vector3f beta = Vector3f(1.0);
  auto     ray  = r;
  int      bounces{0};
  bool     specular{false};

  Vector3f p    = r.m_o;
  Float    rate = 1.0;
  // \sum P(p_n) as a *vector*
  for (bounces = 0;; ++bounces) {
    SInteraction isect;

    bool find_isect = scene.intersect(ray, isect);
    if (find_isect) {
      rate += (isect.m_p - p).norm();
      p = isect.m_p;
    }

    // Handle albedo and normal
    if (bounces == 0) {
      if (find_isect) {
        if (normal != nullptr) *normal = isect.m_ns;
        if (albedo != nullptr)
          *albedo = isect.m_primitive->getMaterial()->getAlbedo();
      } else {
        // no intersection
        if (normal != nullptr) *normal = -ray.m_d;
        if (albedo != nullptr) {
          *albedo = Vector3f(0.0);
          for (const auto &light : scene.m_infLight)
            *albedo += beta * light->Le(ray);
        }
      }
    }

    if (bounces == 0 || specular) {
      if (find_isect) {
        L += beta * isect.Le(-ray.m_d);
      } else {
        // environment light
        for (const auto &light : scene.m_infLight) {
          L += beta * light->Le(ray);
        }
      }
    }

    if (!find_isect || bounces >= m_maxDepth) {
      break;
    }

    // The behavior is quite a bit non-trivial here. I'm not considering ray
    // spawning from image plane for now. Those rays spawned from interaction
    // will carry *m_volume if currently the ray is traveling inside any
    // volume. To be precise, scene.intersect() will inherit primitive's
    // implementation, that is, it will set the correct ray.tMax.
    // ray(ray.tMax) will either reside on the boundary of the volume or any
    // boundary of any object inside the volume(iff the volume exists). To
    // behave correct with multiple volumes, volumes' surfaces will be meshed
    // and own their spaces in BVH. So the property is retained.
    bool         success = false;
    VInteraction vit;
    if (ray.m_volume != nullptr)
      beta *= ray.m_volume->sample(ray, sampler, vit, success);  // TODO
    // if we are sampling the volume
    if (success) {
      L += beta * UniformSampleOneLight(vit, scene, sampler);

      Vector3f wi;
      HGSampleP(vit.m_wo, wi, sampler.get1D(), sampler.get1D(), vit.m_g);

      if (bounces >= m_maxDepth || beta.isZero()) break;
      ray = vit.SpawnRay(wi);

      continue;
    }  // else, we're sampling the surface

    // Handling intersection with specular material
    if (!isect.m_primitive->getMaterial()->isDelta()) {
      // consider the *direct lighting*, i.e. L_e terms in LTE
      L += beta * UniformSampleOneLight(isect, scene, sampler);
    } else {
      specular = true;
    }

    // use the shading normal
    CoordinateTransition trans(isect.m_ns);

    // spawn ray to new direction
    Vector3f wi, wo = -ray.m_d;
    Float    pdf;
    C(isect.m_primitive);
    Vector3f f = isect.m_primitive->getMaterial()->sampleF(
        wo, wi, pdf, sampler.get2D(), Vector2f(0.0), trans);
    if (pdf == 0.0 || f.isZero()) break;

    beta = beta * f * abs(Dot(wi, isect.m_ns)) / pdf;
    ray  = isect.SpawnRay(wi);
  }

  return L;
}

FLG_NAMESPACE_END
