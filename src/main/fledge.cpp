#include "fledge.h"

#include <iostream>
#include <memory>

#include "render.hpp"
#include "scene.hpp"

using namespace fledge;

// LHS coordinate system
int main() {
  // 1. load scene anyway
  //    scene is expected to specify the parameters of camera,
  //    film, integrator, so it is loaded first. Later, render
  //    will initialize integrator with the scene config
  // 2. create the render with scene
  // 3. invoke init() on render
  // 4. invoke render()
  // 5. save the last rendered frame to file
  // auto scene = std::make_unique<Scene>("assets/scene_vol_shape.xml");
  Scene scene("assets/scene_vol_mesh.xml");
  scene.init();

  Render render(&scene);
  SLog("render is created");

  render.init(EBackendType::EOptiXBackend);
  render.preprocess();
  render.render();
  render.saveImage("fledge_out.exr", false);
  scene.m_resource.printStat();
  return 0;
}
