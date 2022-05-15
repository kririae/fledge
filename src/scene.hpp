#ifndef __SCENE_HPP__
#define __SCENE_HPP__

#include <memory>
#include <vector>

#include "fwd.hpp"

SV_NAMESPACE_BEGIN

// scene description
// all resources needed in the process of rendering
// can be loaded from json file or initialized by API
class Scene {
public:
  Scene();
  Scene(const std::string &filename);
  ~Scene() = default;

  bool intersect(const Ray &ray, SInteraction &isect) const;
  AABB getBound() const;

  bool loadFromXml(const std::string &filename);

  int m_resX, m_resY, m_SPP, m_maxDepth;

  std::shared_ptr<Primitive> m_accel;
  std::shared_ptr<Camera>    m_camera;
  std::shared_ptr<Film>      m_film;
  std::shared_ptr<Volume>    m_volume;

  std::vector<std::shared_ptr<Light>> m_light;
  std::vector<std::shared_ptr<Light>> m_infLight;
};

SV_NAMESPACE_END

#endif
