#include "fledge.h"

#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/openvdb.h>

#include <iostream>
#include <memory>

#include "common/vector.h"
#include "debug.hpp"
#include "interaction.hpp"
#include "light.hpp"
#include "primitive.hpp"
#include "render.hpp"
#include "rng.hpp"
#include "scene.hpp"
#include "volume.hpp"

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
  // Render render(std::make_shared<Scene>("assets/scene_basic.xml"));
  Scene  scene("assets/scene_vol_shape.xml");
  Render render(&scene);
  SLog("render is created");

  render.init();
  render.preprocess();
  render.render();
  render.saveImage("fledge_out.exr", false);
  return 0;
}
