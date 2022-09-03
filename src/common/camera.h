#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "common/ray.h"

FLG_NAMESPACE_BEGIN

class Camera {
public:
  F_CPU_GPU Camera(const Vector3f &pos, const Vector3f &lookAt,
                   const Vector3f &up = Vector3f{0, 1, 0}, Float fov = 30)
      : m_pos(pos),
        m_forward(Normalize(lookAt - pos)),
        m_up(up),
        m_right(Normalize(m_up.cross(m_forward))),
        m_fov(fov) {
    C(m_up, m_right);
  }

  F_CPU_GPU Ray generateRay(Float x, Float y, int width, int height) {
    // (pos - center_of_image_plane).norm() is fixed to 1
    Float    aspect_ratio = static_cast<Float>(width) / height;
    Float    fov_ratio    = tanf(PI * m_fov / 360);
    Float    y_offset     = fov_ratio * ((y / height) * 2 - 1.0);
    Float    x_offset     = fov_ratio * aspect_ratio * ((x / width) * 2 - 1.0);
    Vector3f ray_dir      = m_forward + x_offset * m_right +
                       y_offset * Normalize(m_forward.cross(m_right));
    return {m_pos, Normalize(ray_dir)};
  }

  Vector3f m_pos, m_forward, m_up, m_right;
  Float    m_fov;
};

FLG_NAMESPACE_END

#endif