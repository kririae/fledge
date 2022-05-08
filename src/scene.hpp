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
#include "texture.hpp"

SV_NAMESPACE_BEGIN

// scene description
// all resources needed in the process of rendering
// can be loaded from json file or initialized by API
struct Scene {
  Scene() {
    /*
    // objects
    auto sphere_1 = std::make_shared<Sphere>(Vector3f{0.0, 0.5, 0.0}, 0.5);
    auto sphere_2 = std::make_shared<Sphere>(Vector3f{0.0, 2.3, 0.0}, 0.15);
    auto sphere_3 = std::make_shared<Sphere>(Vector3f{0.0, -100.0, 0.0}, 100.0);

    // textures
    auto env_texture =
        std::make_shared<ImageTexture>("assets/venice_sunset_4k.exr");

    // lights
    auto areaLight =
        std::make_shared<AreaLight>(sphere_2, Vector3f::Constant(100.0));
    // auto infAreaLight =
    //     std::make_shared<InfiniteAreaLight>(Vector3f::Constant(0.2));
    auto infAreaLight = std::make_shared<InfiniteAreaLight>(env_texture);

    auto diffuse = std::make_shared<DiffuseMaterial>(Vector3f::Constant(1.0));

    m_resX   = 1024;
    m_resY   = 1024;
    m_SPP    = 128;
    m_camera = std::make_shared<Camera>(Vector3f(0, 1, -5), Vector3f(0, 1, 0));
    m_film   = std::make_shared<Film>(m_resX, m_resY);
    m_accel =
        std::make_shared<NaiveAccel>(std::vector<std::shared_ptr<Primitive>>{
            std::make_shared<ShapePrimitive>(sphere_1),
            std::make_shared<ShapePrimitive>(sphere_2, diffuse, areaLight),
            std::make_shared<ShapePrimitive>(sphere_3)});
    m_light    = std::vector<std::shared_ptr<Light>>{areaLight, infAreaLight};
    m_infLight = std::vector<std::shared_ptr<Light>>{infAreaLight}; */

    auto sphere_1 = std::make_shared<Sphere>(Vector3f{0.0, 0.5, 0.0}, 0.5);
    auto env_texture =
        std::make_shared<ImageTexture>("assets/venice_sunset_4k.exr");
    auto infAreaLight = std::make_shared<InfiniteAreaLight>(env_texture);
    auto diffuse = std::make_shared<DiffuseMaterial>(Vector3f::Constant(1.0));
    m_resX       = 1024;
    m_resY       = 1024;
    m_SPP        = 64;
    m_camera = std::make_shared<Camera>(Vector3f(0, 1, -5), Vector3f(0, 1, 0));
    m_film   = std::make_shared<Film>(m_resX, m_resY);
    m_accel =
        std::make_shared<NaiveAccel>(std::vector<std::shared_ptr<Primitive>>{
            std::make_shared<ShapePrimitive>(sphere_1, diffuse)});
    m_light    = std::vector<std::shared_ptr<Light>>{infAreaLight};
    m_infLight = std::vector<std::shared_ptr<Light>>{infAreaLight};
  }

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

  std::vector<std::shared_ptr<Light>> m_light;
  std::vector<std::shared_ptr<Light>> m_infLight;
};

SV_NAMESPACE_END

#endif
