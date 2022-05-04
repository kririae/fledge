#ifndef __SHAPE_HPP__
#define __SHAPE_HPP__

#include "fwd.hpp"
#include "interaction.hpp"
#include "ray.hpp"

SV_NAMESPACE_BEGIN

class Shape {
public:
  Shape()          = default;
  virtual ~Shape() = default;
  // if the return value is true, [isect] is fully initialized
  // else, perform nothing on [isect]
  virtual bool intersect(const Ray &ray, SInteraction &isect) = 0;
};

class Sphere : public Shape {
public:
  Sphere(const Vector3f &p, Float r) : m_p(p), m_r(r) {}
  ~Sphere() override = default;
  bool intersect(const Ray &ray, SInteraction &isect) override {
    const auto &o   = ray.m_o;
    const auto &d   = ray.m_d;
    const auto &p   = m_p;
    const auto &omp = o - p;

    Float A = d.squaredNorm();
    Float B = 2 * d.dot(omp);
    Float C = omp.squaredNorm() - m_r * m_r;
    // \sqrt{b^2 - 4ac}
    Float S = B * B - 4 * A * C;
    if (S < 0) {
      return false;
    }

    S       = sqrt(S);
    Float t = -(B + S) / (2 * A);
    if (t <= 0) {
      return false;
    }

    Vector3f isect_p = ray(t);
    isect.m_p        = isect_p;
    isect.m_n        = (isect_p - m_p) / m_r;
    isect.m_wo       = -ray.m_d;
    isect.m_t        = t;

    return true;
  }

  Vector3f m_p;
  Float    m_r;
};

SV_NAMESPACE_END

#endif