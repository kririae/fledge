#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__

#include "common/aabb.h"
#include "common/vector.h"
#include "fledge.h"
#include "interaction.hpp"
#include "ray.hpp"

FLG_NAMESPACE_BEGIN

// This naive implementation will preserve all its rigid body properties, i.e.,
// scaling is not considered.
// Those transformation can be reduced to sole move and rotate
struct Transform {
#if 0
  static Vector4f toHomogeneous(const Vector3f &x) {
    Vector4f res;
    res.x() = x.x();
    res.y() = x.y();
    res.z() = x.z();
    res.w() = 1.0;  // definition of homogeneous
    return res;
  }
  static Vector3f fromHomogeneous(const Vector4f &x) {
    Vector3f res;
    res.x() = x.x();
    res.y() = x.y();
    res.z() = x.z();
    return res / x.w();  // yet definition of homogeneous
  }
#endif
  Transform() : m_dx{0.0} {}
  Transform(const Vector3f &dx) : m_dx(dx) {}
  Vector3f applyPoint(const Vector3f &p) const {
    return p + m_dx;
  }  // apply
  Vector3f applyNormal(const Vector3f &n) const {
    return n;
  }  // apply
  SInteraction applyInteraction(const Interaction &inter) const {
    SInteraction post_inter;
    post_inter.m_p  = applyPoint(inter.m_p);
    post_inter.m_ng = applyNormal(inter.m_ng);
    post_inter.m_ns = applyNormal(inter.m_ns);
    post_inter.m_wo = applyNormal(inter.m_wo);
    return post_inter;
  }
  AABB applyAABB(const AABB &aabb) const {
    AABB post_aabb;
    post_aabb.m_min = applyPoint(aabb.m_min);
    post_aabb.m_max = applyPoint(aabb.m_max);
    return post_aabb;
  }
  Ray invRay(const Ray &r) const {
    // Consider one rigid body after transformation, intersect with ray r,
    // resulting a point p with normal n. It is *equivelent* to one rigid body
    // before transformation, intersect with t.invRay(r), resulting in a point
    // t.applyPoint(p), t.applyNormal(n)
    return {r.m_o - m_dx, r.m_d, r.m_tMax};
  }

  Vector3f m_dx;
};

FLG_NAMESPACE_END

#endif
