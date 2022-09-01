#ifndef __RENDER_HPP__
#define __RENDER_HPP__

#include <chrono>
#include <memory>

#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

// serves as *coordinator* between integrator and scene
class Render {
public:
  Render(const std::shared_ptr<Scene> &scene);
  void init();
  bool preprocess();
  bool saveImage(const std::string &name, bool denoise = false);
  bool render();

  /**
   * The following functions are designed for the propose of debugging and will
   * not be documented
   */
  Vector3f *getOriginalResultBuffer();

private:
  std::shared_ptr<Scene>      m_scene;
  std::shared_ptr<Integrator> m_integrator;
  bool                        m_init{false};
};

FLG_NAMESPACE_END

#endif
