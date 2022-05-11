#include "render.hpp"

#include "film.hpp"

SV_NAMESPACE_BEGIN

Render::Render(const std::shared_ptr<Scene> &scene) : m_scene(scene) {
  SV_Log("render is created with scene");
}

void Render::init() {
  SV_Log("render is initialized");
  // m_integrator = std::make_shared<SampleIntegrator>();
  m_integrator = std::make_shared<PathIntegrator>();
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
