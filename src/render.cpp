#include "render.hpp"

#include <openvdb/openvdb.h>

#include <memory>

#include "film.hpp"
#include "integrator.hpp"
#include "scene.hpp"

SV_NAMESPACE_BEGIN

Render::Render(const std::shared_ptr<Scene> &scene) : m_scene(scene) {
  SV_Log("render is created with scene");
}

void Render::init() {
  SV_Log("OpenVDB is ready");
  openvdb::initialize();
  SV_Log("render is ready");
  // m_integrator = std::make_shared<SampleIntegrator>();
  // m_integrator = std::make_shared<PathIntegrator>();
  m_integrator = std::make_shared<SVolIntegrator>();
  m_init       = true;
}

bool Render::preprocess() {
  TODO();
  return false;
}

bool Render::saveImage(const std::string &name) {
  if (!m_init) return false;
  SV_Log("saveImage(%s) is called in render", name.c_str());
  m_scene->m_film->saveImage(name);
  return true;
}

bool Render::render() {
  if (!m_init) return false;
  auto start = std::chrono::high_resolution_clock::now();
  // call Integrator
  m_integrator->render(*m_scene);
  auto                          end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  SV_Log("rendering takes %.9lf s to finish", diff.count());
  return true;
}

SV_NAMESPACE_END
