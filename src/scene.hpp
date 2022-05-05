#ifndef __SCENE_HPP__
#define __SCENE_HPP__

#include <memory>
#include <vector>

#include "accel.hpp"
#include "camera.hpp"
#include "film.hpp"
#include "fwd.hpp"
#include "primitive.hpp"

SV_NAMESPACE_BEGIN

// scene description
// all resources needed in the process of rendering
// can be loaded from json file or initialized by API
class Scene {
public:
  Scene() {
    auto sphere_1 = std::make_shared<Sphere>(Vector3f{0.0, 0.0, 0.0}, 0.5);
    auto sphere_2 = std::make_shared<Sphere>(Vector3f{0.0, 0.3, 1.0}, 0.5);
    m_resX        = 1024;
    m_resY        = 1024;
    m_SPP         = 32;
    m_camera = std::make_shared<Camera>(Vector3f(0, 0, -5), Vector3f(0, 0, 0));
    m_film   = std::make_shared<Film>(m_resX, m_resY);
    m_accel =
        std::make_shared<NaiveAccel>(std::vector<std::shared_ptr<Primitive>>{
            std::make_shared<ShapePrimitive>(sphere_1),
            std::make_shared<ShapePrimitive>(sphere_2)});
  }

  int                        m_resX, m_resY, m_SPP;
  std::shared_ptr<Primitive> m_accel;
  std::shared_ptr<Camera>    m_camera;
  std::shared_ptr<Film>      m_film;

private:
};

SV_NAMESPACE_END

#endif
