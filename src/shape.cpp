#include "shape.hpp"

#include "aabb.hpp"
#include "fwd.hpp"
#include "interaction.hpp"
#include "utils.hpp"

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
  Interaction isect;
  auto        dir = UniformSampleSphere(u);
  auto        p   = m_p + (m_r + NORMAL_EPS) * dir;

  pdf        = 1 / area();
  isect.m_p  = p;
  isect.m_ns = isect.m_ng = dir;
  return isect;
}

AABB Sphere::getBound() const {
  auto v_r = Vector3f::Constant(m_r);
  return AABB(m_p - v_r, m_p + v_r);
}

bool Triangle::intersect(const Ray &ray, Float &tHit, SInteraction &isect) {
  // notice that this `intersect` will consider ray.t_Max
  // but will not modify it
  // Return no intersection if triangle is degenerate

  // the naive intersection code is copied from Assignment in GAMES101
  if (area() == 0) return {};

  Vector3f E1 = m_p1 - m_p0;
  Vector3f E2 = m_p2 - m_p0;
  Vector3f P  = ray.m_d.cross(E2);
  Float    D  = E1.dot(P);
  if (D == 0 || D < 0) return false;

  Vector3f T = ray.m_o - m_p0;
  Float    u = T.dot(P);
  if (u < 0 || u > D) return false;

  Vector3f Q = T.cross(E1);
  Float    v = ray.m_d.dot(Q);
  if (v < 0 || u + v > D) return false;

  float invDet = 1 / D;

  Float tNear = E2.dot(Q) * invDet;
  if (tNear >= ray.m_tMax) return false;

  // HIT
  isect.m_p  = ray(tHit);
  isect.m_wo = -ray.m_d;
  tHit       = tNear;

  u *= invDet;
  v *= invDet;

  return true;
}

Float Triangle::area() const {
  return (m_p1 - m_p0).cross(m_p2 - m_p0).norm();
}

Interaction Triangle::sample(const Vector2f &u, Float &pdf) const {
  TODO();
}

AABB Triangle::getBound() const {
  auto a_ = AABB(m_p0, m_p1);
  auto b_ = AABB(m_p0, m_p2);
  return a_.merge(b_);
}

SV_NAMESPACE_END
