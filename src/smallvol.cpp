#include <Eigen/Dense>
#include <iostream>

#include "camera.hpp"
#include "film.hpp"
#include "fwd.hpp"
#include "interaction.hpp"
#include "render.hpp"
#include "rng.hpp"
#include "shape.hpp"

using Eigen::Vector3d;
using namespace SmallVolNS;

// LHS coordinate system
int main() {
  constexpr int x = 512;
  constexpr int y = 512;
  Render        render(x, y);

  render.init();
  render.render();
  render.saveImage("smallvol_out.exr");
}
