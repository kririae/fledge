#ifndef __RENDER_HPP__
#define __RENDER_HPP__

#include <cstddef>
#include <memory>

#include "accel.hpp"
#include "camera.hpp"
#include "film.hpp"
#include "fwd.hpp"
#include "rng.hpp"
#include "shape.hpp"

SV_NAMESPACE_BEGIN

class Render {
public:
  Render(int resX, int resY, int SPP = 32)
      : m_resX(resX), m_resY(resY), m_SPP(SPP) {}
  ~Render() = default;
  void init() {
    m_camera = std::make_unique<Camera>(Vector3f(0, 0, -5), Vector3f(1, 0, 0));
    m_film   = std::make_unique<Film>(m_resX, m_resY);
    m_accel  = std::make_unique<NaiveAccel>(std::vector<std::shared_ptr<Shape>>{
        std::make_shared<Sphere>(Vector3f{0.0, 0.0, 0.0}, 0.5),
        std::make_shared<Sphere>(Vector3f{0.0, 0.3, 1.0}, 0.5)});
    m_init   = true;
  }
  bool preprocess() {
    TODO();
    return false;
  }
  bool saveImage(const std::string &name) {
    if (!m_init) return false;
    m_film->saveImage(name);
    return true;
  }
  bool render() {
    if (!m_init) return false;
    Random rd;
// TODO: launch worker threads
#pragma omp parallel for
    for (int i = 0; i < m_resX; ++i) {
      for (int j = 0; j < m_resY; ++j) {
        SInteraction isect;
        Vector3f     color = Vector3f::Zero();
        for (int s = 0; s < m_SPP; ++s) {
          auto uv = rd.get2D();
          auto ray =
              m_camera->generateRay(i + uv.x(), j + uv.y(), m_resX, m_resY);
          if (m_accel->intersect(ray, isect)) {
            color += (isect.m_n + Vector3f::Constant(1.0)) / 2;
          } else {
            color += Vector3f::Constant(0.01);
          }
        }

        m_film->getPixel(i, j) = color / m_SPP;
      }
    }

    return true;
  }

private:
  // Scene
  int                     m_resX, m_resY, m_SPP;
  std::unique_ptr<Camera> m_camera;
  std::unique_ptr<Film>   m_film;
  std::unique_ptr<Accel>  m_accel;
  bool                    m_init{false};
};

SV_NAMESPACE_END

#endif
