#ifndef __RENDER_HPP__
#define __RENDER_HPP__

#include <chrono>
#include <memory>

#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

/**
 * @brief The Render class is the main interface of the actual renderer, which
 * serves as the coordinator between integrator and scene.
 */
class Render {
public:
  /**
   * @brief Construct a new Render object from an existing scene pointer.
   *
   * @param scene
   */
  Render(Scene *scene);

  /**
   * @brief After accepting the scene pointer, initialize the render object from
   * the scene objects.
   */
  void init(EBackendType backend = EBackendType::ECPUBackend);

  /**
   * @brief Invoke the preprocess function of the objects. This function is
   * intended to be called before `render()`
   *
   * @return true if preprocess succeed
   */
  bool preprocess();

  /**
   * @brief Save the image from film object
   *
   * @param name The file to save the image to
   * @param denoise Invoke the denoiser or not
   */
  bool saveImage(const std::string &name, bool denoise = false);

  /**
   * @brief Start rendering!
   *
   * @return true if render succeed
   */
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
