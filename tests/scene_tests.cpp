#include <gtest/gtest.h>

#include <iostream>
#include <memory>

#include "common/vector.h"
#include "debug.hpp"
#include "film.hpp"
#include "fledge.h"
#include "light.hpp"
#include "primitive.hpp"
#include "render.hpp"
#include "rng.hpp"
#include "scene.hpp"
#include "volume.hpp"

using namespace fledge;

TEST(Scene, WhiteWithDiffusion) {
  Scene scene;

  auto sphere_1 = scene.m_resource.alloc<Sphere>(Vector3f(0.0), 3.0);
  auto mat      = scene.m_resource.alloc<DiffuseMaterial>(Vector3f(0.5));
  scene.m_primitives.clear();
  scene.m_primitives.push_back(
      scene.m_resource.alloc<ShapePrimitive>(sphere_1, mat));

  auto env_texture = std::make_shared<ConstTexture>(1.0);
  scene.m_light.clear();
  scene.m_infLight.clear();
  scene.m_light.push_back(
      scene.m_resource.alloc<InfiniteAreaLight>(env_texture));
  scene.m_infLight.push_back(scene.m_light[scene.m_light.size() - 1]);

  scene.m_resX            = 1280 / 4;
  scene.m_resY            = 720 / 4;
  scene.m_SPP             = 128;
  scene.m_maxDepth        = 16;
  scene.m_FoV             = 30;  // y axis
  scene.m_up              = Vector3f(0, 1, 0);
  scene.m_origin          = Vector3f(0, 7, -15);
  scene.m_target          = Vector3f(0, 0, 0);
  scene.m_volume          = nullptr;
  scene.m_base_dir        = std::filesystem::path(".");
  scene.m_integrator_type = EIntegratorType::EPathIntegrator;

  scene.init();

  Render render(&scene);
  render.init();
  render.preprocess();
  render.render();
  // render.saveImage("white_with_diffusion.exr", false);

  auto film = render.getFilm();

  Vector<double, 3> average(0.0);
  for (int i = 0; i < film.m_resX; ++i)
    for (int j = 0; j < film.m_resY; ++j)
      average +=
          film.getBuffer(i, j, EFilmBufferType::EColor).cast<double, 3>();

  average /= (film.m_resX * film.m_resY);
  EXPECT_NEAR(average.x(), 1, 1e-3);
  EXPECT_NEAR(average.y(), 1, 1e-3);
  EXPECT_NEAR(average.z(), 1, 1e-3);
}
