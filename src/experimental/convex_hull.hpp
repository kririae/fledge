#ifndef __EXPERIMENTAL_CONVEX_HULL_HPP__
#define __EXPERIMENTAL_CONVEX_HULL_HPP__

#include <boost/container/flat_map.hpp>
#include <cstddef>
#include <list>
#include <memory_resource>
#include <vector>

#include "common/vector.h"
#include "debug.hpp"
#include "experimental/base_bvh.hpp"
#include "fledge.h"
#include "fmt/core.h"

FLG_NAMESPACE_BEGIN
namespace experimental {
namespace detail_ {
struct Face {
  size_t index[3];
  bool   enable{true};
};  // struct Face

/**
 * @brief Doubly connected edge list
 * @see https://en.wikipedia.org/wiki/Doubly_connected_edge_list
 */
class DCEL {
public:
  DCEL(size_t nVert, size_t nFaces, std::pmr::memory_resource *mem_resource)
      : faces(nFaces, mem_resource),
        edges(nFaces * 3, mem_resource),
        verts(nVert, mem_resource),
        m_mem_resource(mem_resource) {}
  ~DCEL() = default;
  struct Edge;
  struct Face {
    Edge *edge{nullptr};
    bool  enable{true};
  };  // struct Face
  struct Edge {
    size_t index[2];
    Edge  *prev{nullptr}, *twin{nullptr}, *next{nullptr};
    Face  *face{nullptr};
  };  // struct Edge
  struct Vertex {
    Vector3f p;
    Edge    *edge{nullptr};
  };

  std::pmr::vector<Face>   faces;
  std::pmr::vector<Edge>   edges;
  std::pmr::vector<Vertex> verts;

private:
  std::pmr::memory_resource *m_mem_resource;
};

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
  ConvexHullInstance(InternalTriangleMesh      *mesh,
                     std::pmr::memory_resource *mem_resource)
      : m_mesh(mesh), m_faces(mem_resource), m_mem_resource(mem_resource) {}
  ~ConvexHullInstance() = default;
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

  /**
   * @brief Convert the ConvexHull instance to InternalTriangleMesh*.
   * Corresponding memory is managed by mem_resource
   *
   * @return InternalTriangleMesh*
   */
  InternalTriangleMesh *toITriangleMesh() {
    InternalTriangleMesh *result;
    auto                  allocator =
        std::pmr::polymorphic_allocator<InternalTriangleMesh>{m_mem_resource};
    result        = allocator.new_object<InternalTriangleMesh>();
    result->nInd  = m_faces.size() * 3;
    result->nVert = m_mesh->nVert;
    result->ind   = allocator.allocate_object<int>(result->nInd);
    result->p     = m_mesh->p;

    size_t face_counter = 0;
    for (auto &face : m_faces) {
      for (int i = 0; i < 3; ++i)
        result->ind[face_counter * 3 + i] = face.index[i];
      face_counter++;
    }

    return result;
  }

  /**
   * @brief Convert the ConvexHull from normal mesh and faces representation to
   * DCEL representation in O(nlogn).
   *
   * @return detail_::DCEL
   */
  detail_::DCEL toDCEL() {
    using dEdge   = detail_::DCEL::Edge;
    using dFace   = detail_::DCEL::Face;
    using dVertex = detail_::DCEL::Vertex;
    boost::container::flat_map<std::pair<size_t, size_t>, dEdge *> mapping;

    detail_::DCEL dcel{static_cast<size_t>(m_mesh->nVert), m_faces.size(),
                       m_mem_resource};

    // Initialize Vertices
    for (int i = 0; i < m_mesh->nVert; ++i)
      dcel.verts[i] = dVertex{.p = m_mesh->p[i], .edge = nullptr};

    size_t face_counter = 0;
    for (auto &face : m_faces) {
      for (int i = 0; i < 3; ++i) {
        dcel.edges[face_counter * 3 + i] = dEdge{
            {face.index[i], face.index[(i + 1) % 3]},
            nullptr,
            nullptr,
            nullptr,
            nullptr,
        };
        auto &last_edge = dcel.edges[face_counter * 3 + i];
        mapping[{face.index[i], face.index[(i + 1) % 3]}] = &last_edge;
        dcel.verts[face.index[i]].edge                    = &last_edge;
      }  // add edges
      dcel.faces[face_counter] =
          dFace{&dcel.edges[face_counter * 3], face.enable};
      auto &e1 = dcel.edges[face_counter * 3];
      auto &e2 = dcel.edges[face_counter * 3 + 1];
      auto &e3 = dcel.edges[face_counter * 3 + 2];

      e1.prev = &e3;
      e1.next = &e2;
      e2.prev = &e1;
      e2.next = &e3;
      e3.prev = &e2;
      e3.next = &e1;
      e1.face = e2.face = e3.face = &dcel.faces[face_counter];
      ++face_counter;
    }  // for face

    // Establish the twin mapping
    for (auto &edge : dcel.edges)
      edge.twin = mapping[{edge.index[1], edge.index[0]}];

    // Verification
    for (auto &vert : dcel.verts) {
      if (vert.edge == nullptr) continue;
      assert(dcel.verts[vert.edge->index[0]].p == vert.p);
    }  // for vert
    for (auto &edge : dcel.edges) {
      assert(edge.face != nullptr);
      assert(edge.prev != nullptr);
      assert(edge.twin != nullptr);
      assert(edge.next != nullptr);
      assert(edge.prev->next == &edge);
      assert(edge.twin->twin == &edge);
      assert(edge.next->prev == &edge);
      assert(edge.prev->index[1] == edge.index[0]);
      assert(edge.twin->index[0] == edge.index[1]);
      assert(edge.twin->index[1] == edge.index[0]);
      assert(edge.next->index[0] == edge.index[1]);
    }  // for edge
    for (auto &face : dcel.faces) {
      assert(face.edge != nullptr);
      assert(face.edge->face == &face);
    }  // for face

    return dcel;
  }

  double surfaceArea() { TODO(); }

  // TODO: use pmr resource
  InternalTriangleMesh         *m_mesh;
  std::pmr::list<detail_::Face> m_faces;
  std::pmr::memory_resource    *m_mem_resource;
};
class ConvexHullBuilder {
public:
  ConvexHullBuilder(InternalTriangleMesh      *mesh,
                    std::pmr::memory_resource *mem_resource)
      : m_mesh(mesh), m_mem_resource(mem_resource) {
    assert(m_mesh->nVert <= 10000);
  }  // ctor
  ConvexHullInstance build() {
    const int n_points = m_mesh->nVert;

    ConvexHullInstance ch{m_mesh, m_mem_resource};
    // TODO: hopefully this is the correct order
    ch.m_faces.push_back(detail_::Face{
        {0, 1, 2},
        true
    });
    ch.m_faces.push_back(detail_::Face{
        {2, 1, 0},
        true
    });  // trick to ensure the first point added

    // Add the later points to construct the base CH
    for (int i = 4; i < n_points; ++i) {
      std::vector<std::vector<bool>> edge_mark(
          n_points, std::vector<bool>(n_points, false));

      // if (ch.inside(m_mesh->p[i])) continue;
      Point3f p      = m_mesh->p[i];
      bool    inside = true;
      for (auto it = ch.m_faces.begin(); it != ch.m_faces.end(); ++it) {
        // Since it is ConvexHull, just remove the face
        if (detail_::PositiveSide(*it, m_mesh, p)) {
          inside = false;
          for (int j = 0; j < 3; ++j) {
            size_t s = it->index[j], t = it->index[(j + 1) % 3];
            edge_mark[s][t] = true;
            it->enable      = false;
          }  // for j
        }
      }

      if (inside) continue;
      // Else, traverse the non-visible faces and link the new faces
      std::pmr::list<detail_::Face> new_faces(m_mem_resource);
      for (auto &face : ch.m_faces) {
        if (!face.enable) continue;
        for (int j = 0; j < 3; ++j) {
          size_t s = face.index[j], t = face.index[(j + 1) % 3];
          if (!edge_mark[s][t] && edge_mark[t][s])
            new_faces.push_back(detail_::Face{
                .index = {static_cast<size_t>(i), t, s},
                  .enable = true
            });
        }  // for j
      }    // for face

      ch.m_faces.remove_if([](auto x) -> bool { return !x.enable; });
      // merge the new faces
      ch.m_faces.splice(ch.m_faces.end(), new_faces);
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
