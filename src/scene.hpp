#ifndef __SCENE_HPP__
#define __SCENE_HPP__

#include <memory>
#include <vector>

#include "accel.hpp"
#include "camera.hpp"
#include "film.hpp"
#include "fwd.hpp"
#include "light.hpp"
#include "primitive.hpp"

SV_NAMESPACE_BEGIN

// scene description
// all resources needed in the process of rendering
// can be loaded from json file or initialized by API
class Scene {
public:
  Scene() {
    auto sphere_1 = std::make_shared<Sphere>(Vector3f{0.0, 0.5, 0.0}, 0.5);
    auto sphere_2 = std::make_shared<Sphere>(Vector3f{0.0, 2.0, 0.0}, 0.1);
    auto sphere_3 = std::make_shared<Sphere>(Vector3f{0.0, -200.0, 0.0}, 200.0);
    auto areaLight =
        std::make_shared<AreaLight>(sphere_2, Vector3f::Constant(10.0));
    auto diffuse = std::make_shared<DiffuseMaterial>(Vector3f::Constant(1.0));
    m_resX       = 1024;
    m_resY       = 1024;
    m_SPP        = 32;
    m_camera = std::make_shared<Camera>(Vector3f(0, 1, -5), Vector3f(0, 1, 0));
    m_film   = std::make_shared<Film>(m_resX, m_resY);
    m_accel =
        std::make_shared<NaiveAccel>(std::vector<std::shared_ptr<Primitive>>{
            std::make_shared<ShapePrimitive>(sphere_1),
            std::make_shared<ShapePrimitive>(sphere_2, diffuse, areaLight),
            std::make_shared<ShapePrimitive>(sphere_3)});
    m_light = std::vector<std::shared_ptr<AreaLight>>{areaLight};
  }

  bool intersect(const Ray &ray, SInteraction &isect) const {
    return m_accel->intersect(ray, isect);
  }

  int                        m_resX, m_resY, m_SPP;
  std::shared_ptr<Primitive> m_accel;
  std::shared_ptr<Camera>    m_camera;
  std::shared_ptr<Film>      m_film;

  std::vector<std::shared_ptr<AreaLight>> m_light;

private:
};

SV_NAMESPACE_END

#endif
