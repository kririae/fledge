#include "render.hpp"

#include <openvdb/openvdb.h>

#include <memory>

#include "common/sampler.h"
#include "debug.hpp"
#include "external/oidn/oidn.hpp"
#include "film.hpp"
#include "fledge.h"
#include "integrator.hpp"
#include "light.hpp"
#include "optix/optix_interface.hpp"
#include "scene.hpp"

FLG_NAMESPACE_BEGIN

Render::Render(Scene *scene) : m_scene(scene) {}

void Render::init(EBackendType backend) {
  detail_::GetPrimeList();
  SLog("Global prime list is initialized");

  openvdb::initialize();
  SLog("OpenVDB is ready");

  SLog("Render is initializing");

  if (backend == EBackendType::ECPUBackend) {
    switch (int(m_scene->m_integrator_type)) {
      case int(EIntegratorType::EPathIntegrator):
        m_integrator =
            m_scene->m_resource.alloc<PathIntegrator>(m_scene->m_maxDepth);
        break;
      case int(EIntegratorType::EVolPathIntegrator):
        m_integrator =
            m_scene->m_resource.alloc<VolPathIntegrator>(m_scene->m_maxDepth);
        break;
      default:
        TODO();
    }  // switch integrator_type
  } else if (backend == EBackendType::EOptiXBackend) {
    optix::InitOptiX();
    SLog("Renderer initialized with OptiX backend");
    m_integrator =
        m_scene->m_resource.alloc<VolPathIntegrator>(m_scene->m_maxDepth);
  } else {
    SErr("This backend is currently not supported");
  }

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

  return true;
}

Film &Render::getFilm() {
  return *m_scene->m_film;
}

const Film &Render::getFilm() const {
  return *m_scene->m_film;
}

FLG_NAMESPACE_END
