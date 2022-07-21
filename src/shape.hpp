#ifndef __SHAPE_HPP__
#define __SHAPE_HPP__

#include "fwd.hpp"
#include "plymesh.hpp"

SV_NAMESPACE_BEGIN

class Shape {
public:
  virtual ~Shape() = default;
  // if the return value is true, [isect] is fully initialized
  // else, perform nothing on [isect]
  virtual bool intersect(const Ray &, Float &, SInteraction &) = 0;

  virtual Float       area() const                               = 0;
  virtual Interaction sample(const Vector2f &, Float &pdf) const = 0;

  // Sample a point on the shape given a reference point |ref| and
  // return the PDF with respect to solid angle from |ref|.
  virtual Interaction sample(const Interaction &ref, const Vector2f &u,
                             Float &pdf) const;

  virtual AABB  getBound() const = 0;
  virtual Float pdf(const Interaction &) const;
};

class Sphere : public Shape {
public:
  Sphere(const Vector3f &p, Float r) : m_p(p), m_r(r) {}
  ~Sphere() override = default;
  bool  intersect(const Ray &ray, Float &tHit, SInteraction &isect) override;
  Float area() const override;
  Interaction sample(const Vector2f &u, Float &pdf) const override;
  // defined
  // Interaction sample(const Interaction &ref, const Vector2f &u,
  //                    Float &pdf) const override;
  AABB getBound() const override;
  bool operator==(const Sphere &a) const {
    return m_p == a.m_p && m_r == a.m_r;
  }

  Vector3f m_p;
  Float    m_r;
};

class Triangle : public Shape {
public:
  Triangle(const Vector3f &a, const Vector3f &b, const Vector3f &c)
      : m_p0(a), m_p1(b), m_p2(c) {}

  bool  intersect(const Ray &ray, Float &tHit, SInteraction &isect) override;
  Float area() const override;
  Interaction sample(const Vector2f &u, Float &pdf) const override;
  AABB        getBound() const override;
  bool        operator==(const Triangle &t) const {
           // Order aware
    return m_p0 == t.m_p0 && m_p1 == t.m_p1 && m_p2 == t.m_p2;
  }

  Vector3f m_p0, m_p1, m_p2;
};

SV_NAMESPACE_END

#endif
