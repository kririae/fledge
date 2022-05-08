#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/openvdb.h>

#include <Eigen/Dense>
#include <iostream>
#include <memory>

#include "fwd.hpp"
#include "render.hpp"
#include "rng.hpp"
#include "scene.hpp"

using namespace SmallVolNS;

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
  Render render(std::make_shared<Scene>());

  // render.init();
  // render.render();
  // render.saveImage("smallvol_out.exr");

  openvdb::initialize();

  std::string       filename = "assets/wdas_cloud/wdas_cloud_eighth.vdb";
  openvdb::io::File file(filename);
  Log("load OpenVDB file %s", filename.c_str());
  file.open();
  for (openvdb::io::File::NameIterator nameIter = file.beginName();
       nameIter != file.endName(); ++nameIter) {
    Log("find grid: %s", nameIter.gridName().c_str());
  }

  auto baseGrid = file.readGrid("density");
  auto grid     = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
  for (openvdb::MetaMap::MetaIterator iter = grid->beginMeta();
       iter != grid->endMeta(); ++iter) {
    const std::string     &name          = iter->first;
    openvdb::Metadata::Ptr value         = iter->second;
    std::string            valueAsString = value->str();
    Log("find metadata=(%s, %s)", name.c_str(), valueAsString.c_str());
  }

  file.close();
}
