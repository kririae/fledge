#ifndef __EXPERIMENTAL_CONVEX_HULL_HPP__
#define __EXPERIMENTAL_CONVEX_HULL_HPP__

#include <list>
#include <memory_resource>
#include <vector>

#include "common/vector.h"
#include "experimental/base_bvh.hpp"
#include "fledge.h"
#include "fmt/core.h"

FLG_NAMESPACE_BEGIN
namespace experimental {
namespace detail_ {
struct Face {
  size_t index[3];
};  // struct Face
struct Edge {
  size_t index[2];
  Edge  *prev, *twin, *next;
  Face  *face;
};  // struct Edge
bool SameDirection(const Vector3f &x, const Vector3f &y) {
  return Dot(x, y) > 0;
}
Normal3f Normal(const Vector3f &x, const Vector3f &y, const Vector3f &z) {
  // in LHS coordinate system
  return Normalize(Cross(z - x, y - z));
}
Normal3f Normal(const Face &face, const InternalTriangleMesh *mesh) {
  return Normal(mesh->p[face.index[0]], mesh->p[face.index[1]],
                mesh->p[face.index[2]]);
}
/**
 * @brief Given a directional face and a point, judge if the point is in the
 * positive side of the plane
 *
 * @return true if the point is in the positive side of the plane
 */
bool PositiveSide(const Face &face, const InternalTriangleMesh *mesh,
                  const Point3f &p) {
  return SameDirection(mesh->p[face.index[0]] - p, Normal(face, mesh));
}
}  // namespace detail_
// Intermediate Representation
class ConvexHullInstance {
public:
  // TODO: use pmr resource
  InternalTriangleMesh    *m_mesh;
  std::list<detail_::Face> m_faces;
  std::vector<bool>        m_used;

  /**
   * @brief Judge if the point is inside of the ConvexHull
   *
   * @param p
   * @return true iff the point is inside the CH
   */
  bool inside(const Point3f &p) {
    for (auto &i : m_faces)
      if (detail_::PositiveSide(i, m_mesh, p)) return false;
    return true;
  }
};
class ConvexHullBuilder {
public:
  ConvexHullBuilder(InternalTriangleMesh      *mesh,
                    std::pmr::memory_resource *mem_resource)
      : m_mesh(mesh), m_mem_resource(mem_resource) {}
  ConvexHullInstance build() {
    const int n_points = m_mesh->nVert;

    ConvexHullInstance ch{.m_mesh = m_mesh,
                          .m_used = std::vector<bool>(false, m_mesh->nVert)};
    // TODO: hopefully this is the correct order
    ch.m_faces.push_back(detail_::Face{0, 1, 2});
    ch.m_faces.push_back(
        detail_::Face{2, 1, 0});  // trick to ensure the first point added

    // Add the later points to construct the base CH
    for (int i = 4; i < n_points; ++i) {
      // if (ch.inside(m_mesh->p[i])) continue;
      Point3f p      = m_mesh->p[i];
      bool    inside = true;
      for (auto it = ch.m_faces.begin(); it != ch.m_faces.end(); ++it) {
        // Since it is ConvexHull, just remove the face
        if (detail_::PositiveSide(*it, m_mesh, p)) {
          inside = false;
          ch.m_faces.erase(it);
        }
      }
    }  // main CH loop

    return ch;
  }

private:
  InternalTriangleMesh      *m_mesh;  // naive mesh storage
  std::pmr::memory_resource *m_mem_resource{std::pmr::get_default_resource()};
};
}  // namespace experimental
FLG_NAMESPACE_END

#endif
