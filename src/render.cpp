#include "render.hpp"

#include <openvdb/openvdb.h>

#include <memory>

#include "debug.hpp"
#include "film.hpp"
#include "integrator.hpp"
#include "light.hpp"
#include "scene.hpp"
#include "spec/oidn/oidn.hpp"

FLG_NAMESPACE_BEGIN

Render::Render(const std::shared_ptr<Scene> &scene) : m_scene(scene) {
  SLog("render is created with scene");
}

void Render::init() {
  SLog("OpenVDB is ready");
  openvdb::initialize();
  SLog("render is ready");
  // m_integrator = std::make_shared<SampleIntegrator>();
  m_integrator = std::make_shared<PathIntegrator>(m_scene->m_maxDepth);
  // m_integrator = std::make_shared<SVolIntegrator>();
  m_init = true;
}

bool Render::preprocess() {
  for (auto &light : m_scene->m_infLight) light->preprocess(*m_scene);
  return true;
}

bool Render::saveImage(const std::string &name, bool denoise) {
  if (!m_init) return false;
  auto name_ = m_scene->getPath(name);
  SLog("saveImage(%s)", name_.c_str());
  if (denoise) {
    Film post_filtered = Denoise(*(m_scene->m_film));
    post_filtered.saveBuffer(name_, EFilmBufferType::EOutput);
  } else {
    m_scene->m_film->saveBuffer(name_, EFilmBufferType::EColor);
  }

  return true;
}

bool Render::render() {
  if (!m_init) return false;
  auto start = std::chrono::high_resolution_clock::now();
  // call Integrator
  m_integrator->render(*m_scene);
  auto                          end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  SLog("rendering takes %.9lf s to finish", diff.count());
  SLog("measurement: ");

  return true;
}

FLG_NAMESPACE_END
