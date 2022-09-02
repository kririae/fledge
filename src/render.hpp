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
  Render(Scene *scene);
  void init();
  bool preprocess();
  bool saveImage(const std::string &name, bool denoise = false);
  bool render();

  /**
   * The following functions are designed for the propose of debugging and will
   * not be documented
   */
  Film       &getFilm();
  const Film &getFilm() const;

private:
  Scene      *m_scene{nullptr};
  Integrator *m_integrator{nullptr};
  bool        m_init{false};
};

FLG_NAMESPACE_END

#endif
