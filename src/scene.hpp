#ifndef __SCENE_HPP__
#define __SCENE_HPP__

#include <oneapi/tbb/scalable_allocator.h>

#include <filesystem>
#include <memory>
#include <memory_resource>
#include <string>
#include <vector>

#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"
#include "resource.hpp"

FLG_NAMESPACE_BEGIN

using std::filesystem::path;

/**
 * @brief Scene description for fledge renderer, which is designed to contain
 * all the renderer *resources* and their *relationship* before ResourceManager
 * is introduced. Scene description can be loaded from json file or initialized
 * by API.
 */
class Scene {
public:
  Scene();
  /**
   * @brief Construct a new Scene object with filename of the scene setting in
   * xml currently.
   *
   * @param filename
   */
  Scene(const std::string &filename);
  /**
   * @brief Destroy the Scene object
   *
   */
  ~Scene() = default;

  /**
   * @brief Init the scene from parameter settings reside in scene settings.
   *
   * @return true if init succeed
   */
  bool init();

  /**
   * @brief Intersect the ray with objects in the scene.
   * @see The function primitive.intersect(), which is designed to behave
   * exactly the same as scene.intersect()
   *
   * @param ray The input ray with its origin, direction and m_tMax considered.
   * @param isect The output Intersection, which will be initialized(or
   * modified) iff the function returns true.
   * @return true Considering ray.m_tMax, if there's object in the scene within
   * the range, true is returned.
   */
  bool intersect(const Ray &ray, SInteraction &isect) const;
  /**
   * @brief Get the Bound of the scene represented by AABB.
   *
   * @return AABB the boundary of the scene except for the infinite area light
   */
  AABB getBound() const;

  path        getPath(const path &asset_path);
  std::string getPath(const std::string &asset_path);

  /**
   * The MAIN resource manager
   */
  // std::pmr::memory_resource *m_upstream =
  //     oneapi::tbb::scalable_memory_resource();
  std::pmr::memory_resource *m_upstream =
      GlobalManagedMemoryResource();  // cudaMallocManaged
  Resource m_resource{m_upstream};

  /**
   * The following variables are to be initialized *before* init() by
   * either xml configurations or APIs.
   */
  int             m_resX{0}, m_resY{0}, m_SPP{0}, m_maxDepth{0};
  Float           m_FoV{0};
  Vector3f        m_up{0}, m_origin{0}, m_target{0};
  path            m_base_dir{std::filesystem::path(".")};
  EIntegratorType m_integrator_type{EIntegratorType::EPathIntegrator};

  std::vector<Primitive *> m_primitives{};
  std::vector<Light *>     m_light{};
  std::vector<Light *>     m_infLight{};
  Volume                  *m_volume{nullptr};

  /**
   * The following variables are to be initialized *after*(by) init()
   */
  Primitive *m_accel{nullptr};
  Camera    *m_camera{nullptr};
  Film      *m_film{nullptr};

private:
  void parseXML(const std::string &filename);
};

FLG_NAMESPACE_END

#endif
