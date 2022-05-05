#ifndef __RENDER_HPP__
#define __RENDER_HPP__

#include <memory>

#include "fwd.hpp"
#include "integrator.hpp"
#include "rng.hpp"
#include "scene.hpp"

SV_NAMESPACE_BEGIN

// serves as *coordinator* between integrator and scene
class Render {
public:
  Render(const std::shared_ptr<Scene> &scene) : m_scene(scene) {
    Log("render is created with scene");
  }
  ~Render() = default;
  void init() {
    Log("render is initialized");
    // m_integrator = std::make_shared<SampleIntegrator>();
    m_integrator = std::make_shared<PathIntegrator>();
    m_init       = true;
  }

  bool preprocess() {
    TODO();
    return false;
  }

  bool saveImage(const std::string &name) {
    if (!m_init) return false;
    Log("saveImage(%s) is called in render", name.c_str());
    m_scene->m_film->saveImage(name);
    return true;
  }

  bool render() {
    if (!m_init) return false;
    Log("rendering process started from render");
    // call Integrator
    m_integrator->render(*m_scene);
    return true;
  }

private:
  std::shared_ptr<Scene>      m_scene;
  std::shared_ptr<Integrator> m_integrator;
  bool                        m_init{false};
};

SV_NAMESPACE_END

#endif
