#ifndef __CAMERA_HPP__
#define __CAMERA_HPP__

#include "fwd.hpp"
#include "ray.hpp"

SV_NAMESPACE_BEGIN

class Camera {
public:
  Camera(const Vector3f &pos, const Vector3f &lookAt,
         const Vector3f &up = Vector3f{0, 1, 0}, Float fov = 30)
      : m_pos(pos),
        m_forward((lookAt - pos).normalized()),
        m_up(up),
        m_right(m_up.cross(m_forward)),
        m_fov(fov) {
    assert(m_up.norm() == 1);
    assert(m_right.norm() == 1);
    m_forward.normalize();
  }

  Ray generateRay(Float x, Float y, int width, int height) {
    Float    aspect_ratio = static_cast<Float>(width) / height;
    Float    tmp          = tanf(PI * m_fov / 360);
    Float    y_offset     = tmp * ((y / height) * 2 - 1.0);
    Float    x_offset     = tmp * aspect_ratio * ((x / width) * 2 - 1.0);
    Vector3f ray_dir      = m_forward + x_offset * m_right + y_offset * m_up;
    return {m_pos, ray_dir.normalized()};
  }

  Vector3f m_pos, m_forward, m_up, m_right;
  Float    m_fov;
};

SV_NAMESPACE_END

#endif