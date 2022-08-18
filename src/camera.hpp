#ifndef __CAMERA_HPP__
#define __CAMERA_HPP__

#include "debug.hpp"
#include "fwd.hpp"
#include "ray.hpp"
#include "vector.hpp"

SV_NAMESPACE_BEGIN

class Camera {
public:
  Camera(const Vector3f &pos, const Vector3f &lookAt,
         const Vector3f &up = Vector3f{0, 1, 0}, Float fov = 30)
      : m_pos(pos),
        m_forward((lookAt - pos).stableNormalized()),
        m_up(up),
        m_right((m_up.cross(m_forward)).stableNormalized()),
        m_fov(fov) {
    assert(m_up.norm() == 1);
    assert(m_forward.norm() == 1);
    assert(m_right.norm() == 1);
  }

  Ray generateRay(Float x, Float y, int width, int height) {
    // (pos - center_of_image_plane).norm() is fixed to 1
    Float    aspect_ratio = static_cast<Float>(width) / height;
    Float    fov_ratio    = tanf(PI * m_fov / 360);
    Float    y_offset     = fov_ratio * ((y / height) * 2 - 1.0);
    Float    x_offset     = fov_ratio * aspect_ratio * ((x / width) * 2 - 1.0);
    Vector3f ray_dir      = m_forward + x_offset * m_right +
                       y_offset * (m_forward.cross(m_right).stableNormalized());
    return {m_pos, ray_dir.stableNormalized()};
  }

  Vector3f m_pos, m_forward, m_up, m_right;
  Float    m_fov;
};

SV_NAMESPACE_END

#endif