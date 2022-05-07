#include "shape.hpp"

#include "fwd.hpp"
#include "interaction.hpp"

SV_NAMESPACE_BEGIN

Float Shape::pdf(const Interaction &) const {
  return 1 / area();
}

// Sample a point on the shape given a reference point |ref| and
// return the PDF with respect to solid angle from |ref|.
Interaction Shape::sample(const Interaction &ref, const Vector2f &u,
                          Float &pdf) const {
  Interaction intr   = sample(u, pdf);
  Vector3f    wi     = intr.m_p - ref.m_p;
  Float       s_norm = wi.squaredNorm();
  if (s_norm == 0) {
    pdf = 0.0;
  } else {
    wi.normalize();
    // dw = dA cosTheta/|d|^2
    // p(w) = p(A) |d|^2/cosTheta
    pdf *= s_norm / intr.m_ng.dot(-wi);
  }

  return intr;
}

bool Sphere::intersect(const Ray &ray, Float &tHit, SInteraction &isect) {
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
  if (t <= 0 || t > ray.m_tMax) {
    return false;
  }

  Vector3f isect_p = ray(t);
  isect.m_p        = isect_p;
  isect.m_ns = isect.m_ng = (isect_p - m_p) / m_r;
  isect.m_wo              = -ray.m_d;
  tHit                    = t;

  return true;
}

Float Sphere::area() const {
  return 4 * PI * m_r * m_r;
}

Interaction Sphere::sample(const Vector2f &u, Float &pdf) const {
  SInteraction isect;
  auto         dir = UniformSampleSphere(u);
  auto         p   = m_p + (m_r + NORMAL_EPS) * dir;

  pdf        = 1 / area();
  isect.m_p  = p;
  isect.m_ns = isect.m_ng = dir;
  return isect;
}

SV_NAMESPACE_END
