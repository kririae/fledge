#include <gtest/gtest.h>

#include <memory>

#include "fledge.h"
#include "light.hpp"
#include "material.hpp"
#include "primitive.hpp"
#include "render.hpp"
#include "scene.hpp"
#include "shape.hpp"
#include "texture.hpp"

using namespace fledge;

static std::shared_ptr<Scene> makeWhiteWithDiffusion() {
  std::shared_ptr<Scene> scene = std::make_shared<Scene>();

  auto sphere      = std::make_shared<Sphere>(Vector3f{0.0, 0.0, 0.0}, 3.0);
  auto env_texture = std::make_shared<ConstTexture>(1.0);
  std::shared_ptr<Light> inf_area_light =
      std::make_shared<InfiniteAreaLight>(env_texture);
  auto mat = std::make_shared<DiffuseMaterial>(1.0);

  scene->m_resX     = 1280 / 4;
  scene->m_resY     = 720 / 4;
  scene->m_SPP      = 128;
  scene->m_maxDepth = 16;
  scene->m_FoV      = 30;  // y axis
  scene->m_up       = Vector3f(0, 1, 0);
  scene->m_origin   = Vector3f(0, 7, -15);
  scene->m_target   = Vector3f(0);
  scene->m_light.push_back(inf_area_light);
  scene->m_infLight.push_back(inf_area_light);
  scene->m_primitives.push_back(std::make_shared<ShapePrimitive>(sphere, mat));

  scene->init();

  return scene;
}

TEST(Scene, WhiteWithDiffusion) {
  auto   scene = makeWhiteWithDiffusion();
  Render render(scene);
  render.init();
  render.preprocess();
  render.render();

  Vector3f *buffer = render.getOriginalResultBuffer();
}
