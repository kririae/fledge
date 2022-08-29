#ifndef __SHAPE_HPP__
#define __SHAPE_HPP__

#include <memory>

#include "fledge.h"
#include "debug.hpp"
#include "plymesh.hpp"
#include "common/vector.h"

FLG_NAMESPACE_BEGIN

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

// Following the design of PBRT
// https://pbr-book.org/3ed-2018/Shapes/Triangle_Meshes
struct TriangleMesh {
  // assume that n := nTriangles
  // nInd = 3*n
  // nVert = ???
  int nInd, nVert;

  std::unique_ptr<int[]>      ind;
  std::unique_ptr<Vector3f[]> p, n{};
  std::unique_ptr<Vector2f[]> uv;
};

class Triangle : public Shape {
public:
  Triangle(std::shared_ptr<TriangleMesh> mesh, int *v) : m_mesh(mesh), m_v(v) {}
  // Triangle Mesh can be indexed with index of triangle, where
  // ind[idx], ind[idx+1], ind[idx+2] will point to three positions of the
  // triangle
  Triangle(std::shared_ptr<TriangleMesh> mesh, int idx)
      : m_mesh(mesh), m_v(mesh->ind.get() + 3 * idx) {}

  bool  intersect(const Ray &ray, Float &tHit, SInteraction &isect) override;
  Float area() const override;
  Interaction sample(const Vector2f &u, Float &pdf) const override;
  AABB        getBound() const override;
  bool        operator==(const Triangle &t) const {
           // Order aware
    const auto p0   = m_mesh->p[*m_v];
    const auto p1   = m_mesh->p[*(m_v + 1)];
    const auto p2   = m_mesh->p[*(m_v + 2)];
    const auto t_p0 = t.m_mesh->p[*t.m_v];
    const auto t_p1 = t.m_mesh->p[*(t.m_v + 1)];
    const auto t_p2 = t.m_mesh->p[*(t.m_v + 2)];
    return p0 == t_p0 && p1 == t_p1 && p2 == t_p2;
  }

  std::shared_ptr<TriangleMesh> m_mesh;
  int                          *m_v;
};

FLG_NAMESPACE_END

#endif
