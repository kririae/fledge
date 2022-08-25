#ifndef __SCENE_HPP__
#define __SCENE_HPP__

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "fwd.hpp"
#include "vector.hpp"

FLG_NAMESPACE_BEGIN

using std::filesystem::path;
// scene description
// all resources needed in the process of rendering
// can be loaded from json file or initialized by API
class Scene {
public:
  Scene();
  Scene(const std::string &filename);
  ~Scene() = default;

  // init scene from parameter settings
  bool init();
  bool intersect(const Ray &ray, SInteraction &isect) const;
  AABB getBound() const;

  static Scene parseXML(const std::string &filename);
  path         getPath(const path &asset_path);
  std::string  getPath(const std::string &asset_path);

  // Before init()
  int      m_resX, m_resY, m_SPP, m_maxDepth;
  Float    m_FoV;
  Vector3f m_up, m_origin, m_target;

  std::vector<std::shared_ptr<Primitive>> m_primitives;

  // After init()
  std::shared_ptr<Primitive> m_accel;
  std::shared_ptr<Camera>    m_camera;
  std::shared_ptr<Film>      m_film;
  std::shared_ptr<Volume>    m_volume;

  std::vector<std::shared_ptr<Light>> m_light;
  std::vector<std::shared_ptr<Light>> m_infLight;

private:
  path m_base_dir;
};

FLG_NAMESPACE_END

#endif
