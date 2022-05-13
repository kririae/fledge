#ifndef __SCENE_HPP__
#define __SCENE_HPP__

#include <memory>
#include <vector>

#include "accel.hpp"
#include "fwd.hpp"

SV_NAMESPACE_BEGIN

class Film;
class Light;
class Camera;
class Volume;
class Primitive;
// scene description
// all resources needed in the process of rendering
// can be loaded from json file or initialized by API
struct Scene {
  Scene();
  Scene(const std::string &filename) {
    bool success = loadFromXml(filename);
    assert(success);
  }

  bool intersect(const Ray &ray, SInteraction &isect) const {
    return m_accel->intersect(ray, isect);
  }

  bool loadFromXml(const std::string &filename);

  int                        m_resX, m_resY, m_SPP, m_maxDepth;
  std::shared_ptr<Primitive> m_accel;
  std::shared_ptr<Camera>    m_camera;
  std::shared_ptr<Film>      m_film;
  std::shared_ptr<Volume>    m_volume;

  std::vector<std::shared_ptr<Light>> m_light;
  std::vector<std::shared_ptr<Light>> m_infLight;
};

SV_NAMESPACE_END

#endif
