#ifndef __EXPERIMENTAL_CONVEX_HULL_HPP__
#define __EXPERIMENTAL_CONVEX_HULL_HPP__

#include <boost/container/flat_map.hpp>
#include <cassert>
#include <cstddef>
#include <limits>
#include <list>
#include <memory_resource>
#include <numeric>
#include <utility>
#include <variant>
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
struct Edge {
  size_t index[2];
};  // struct Edge

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
  return Normalize(Cross(y - x, z - x));
}
Normal3f Normal(const Face &face, const Vector3f *points_array) {
  return Normal(points_array[face.index[0]], points_array[face.index[1]],
                points_array[face.index[2]]);
}
/**
 * @brief Given a directional face and a point, judge if the point is in the
 * positive side of the plane
 *
 * @return true if the point is in the positive side of the plane
 */
bool PositiveSide(const Face &face, const Vector3f *points_array,
                  const Point3f &p) {
  return SameDirection(points_array[face.index[0]] - p,
                       Normal(face, points_array));
}
Vector3f ProjectPointToLine(const Vector3f &p, const Vector3f &a,
                            const Vector3f &b) {
  const Vector3f &dir = Normalize(b - a);
  float           len = Dot(p - a, dir);
  return a + len * dir;
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
      if (detail_::PositiveSide(i, m_mesh->p, p)) return false;
    return true;
  }

  void assumeConvex() {
    assert(m_mesh->nInd % 3 == 0);
    m_faces.clear();
    for (int i = 0; i < m_mesh->nInd; i += 3)
      m_faces.push_back(detail_::Face{
          {static_cast<size_t>(m_mesh->ind[i]),
           static_cast<size_t>(m_mesh->ind[i + 1]),
           static_cast<size_t>(m_mesh->ind[i + 2])}
      });
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

  bool verifyOrientation() {
    bool result = true;

    std::vector<int> visit(m_mesh->nVert, 0);
    // Assume it is convex
    for (auto &face : m_faces)
      for (int i = 0; i < 3; ++i) visit[face.index[i]] = 1;

    Vector3f    center{0};
    std::size_t cnt = 0;
    for (int i = 0; i < m_mesh->nVert; ++i) {
      if (visit[i]) {
        center += m_mesh->p[i];
        ++cnt;
      }
    }  // for i
    center /= cnt;

    for (auto &face : m_faces) {
      Vector3f a          = m_mesh->p[face.index[0]];
      Vector3f b          = m_mesh->p[face.index[1]];
      Vector3f c          = m_mesh->p[face.index[2]];
      Vector3f abc_center = (a + b + c) / 3;
      result &=
          detail_::SameDirection(abc_center - center, detail_::Normal(a, b, c));
    }

    return result;
  }

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
    for (int i = 3; i < n_points; ++i) {
      std::vector<std::vector<bool>> edge_mark(
          n_points, std::vector<bool>(n_points, false));

      // if (ch.inside(m_mesh->p[i])) continue;
      Point3f p      = m_mesh->p[i];
      bool    inside = true;
      for (auto it = ch.m_faces.begin(); it != ch.m_faces.end(); ++it) {
        // Since it is ConvexHull, just remove the face
        if (!detail_::PositiveSide(*it, m_mesh->p, p)) {
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
                .index = {t, s, static_cast<size_t>(i)},
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

namespace detail_ {
struct SimplexBase {
  virtual ~SimplexBase() = default;

  virtual SimplexBase *promote(const Vector3f &p) const = 0;
  virtual std::size_t  getSize() const                  = 0;
};

template <std::size_t N>
requires(N <= 3) struct Simplex : public SimplexBase {
};

template <>
struct Simplex<3> : public SimplexBase {
  static constexpr std::size_t size = 3;

  Vector3f points[4];
  Face     faces[4];
  bool     inside(const Vector3f &p) const {
        // format error
    bool side = PositiveSide(faces[0], points, p);
    for (std::size_t i = 1; i < 4; ++i)
      if (PositiveSide(faces[i], points, p) != side) return false;
    return true;
  }

  SimplexBase *promote(const Vector3f &p) const override { TODO(); }
  std::size_t  getSize() const override { return size; }
};

template <>
struct Simplex<2> : public SimplexBase {
  static constexpr std::size_t size = 2;

  Vector3f points[3];
  Face     face;

  // retain the default ctor
  Simplex() = default;
  /**
   * @brief Construct a new Simplex object from face and points_array
   */
  Simplex(const Face &face_, const Vector3f points_array[4]) {
    for (int i = 0; i < 3; ++i) points[i] = points_array[face_.index[i]];
    for (int i = 0; i < 3; ++i) face.index[i] = i;
    assert(Normal(face, points) == Normal(face_, points_array));
  }  // Simplex ctor

  std::size_t getSize() const override { return size; }
  Simplex<3> *promote(const Vector3f &p) const override {
    Simplex<3> result;
    for (int i = 0; i < 3; ++i) result.points[i] = points[i];
    result.points[3] = p;
    result.faces[0]  = face;

    // reorient the face
    if (PositiveSide(face, points, p))
      std::swap(result.faces[0].index[0], result.faces[0].index[1]);

    for (int i = 0; i < 3; ++i) {
      auto  A        = face.index[i];
      auto  B        = face.index[(i + 1) % 3];
      auto  C        = 3;
      auto &face_    = result.faces[i + 1];
      face_.index[0] = A;
      face_.index[1] = C;
      face_.index[2] = B;
    }

    // TODO: do not deallocate for now
    Simplex<3> *ret = new Simplex<3>(result);
    return ret;
  }
};

template <>
struct Simplex<1> : public SimplexBase {
  static constexpr std::size_t size = 1;

  Vector3f    points[2];
  std::size_t getSize() const override { return size; }
  Simplex<2> *promote(const Vector3f &p) const override {
    Simplex<2> result;
    result.points[0] = points[0];
    result.points[1] = points[1];
    result.points[2] = p;
    result.face      = Face{
             .index = {0, 1, 2},
               .enable = true
    };

    Simplex<2> *ret = new Simplex<2>(result);
    return ret;
  }
};

template <>
struct Simplex<0> : public SimplexBase {
  static constexpr std::size_t size = 0;

  Simplex() = default;
  Simplex(const Vector3f &point_) : point(point_) {}

  Vector3f    point;
  std::size_t getSize() const override { return size; }
  Simplex<1> *promote(const Vector3f &p) const override {
    Simplex<1> result;
    result.points[0] = point;
    result.points[1] = p;

    Simplex<1> *ret = new Simplex<1>(result);
    return ret;
  }
};

inline float DistanceToFace(const Face &face, const Vector3f *points_array,
                            const Vector3f &p) {
  const Vector3f &a = points_array[face.index[0]];
  const Normal3f &n = Normal(face, points_array);
  return abs(Dot(p - a, n));
}

inline Vector3f Support(const ConvexHullInstance &shape, const Vector3f &d) {
  std::pair<Vector3f, float> result =
      std::make_pair(Vector3f{}, -std::numeric_limits<float>::max());
  for (auto &face : shape.m_faces) {
    for (int j = 0; j < 3; ++j) {
      const Vector3f &p           = shape.m_mesh->p[face.index[j]];
      auto            dot_product = Dot(p, d);
      if (dot_product > result.second) result = std::make_pair(p, dot_product);
    }
  }

  assert(result.second != -std::numeric_limits<float>::max());
  return result.first;
}

inline std::tuple<SimplexBase *, Vector3f, bool> NearestSimplex(
    SimplexBase *s) {
  std::size_t N = s->getSize();
  if (N == 1) {
    const Simplex<1> *s_ = dynamic_cast<const Simplex<1> *>(s);
    return {s, -ProjectPointToLine(Vector3f{}, s_->points[0], s_->points[1]),
            false};
  } else if (N == 2) {
    const Simplex<2> *s_ = dynamic_cast<const Simplex<2> *>(s);
    Normal3f          n  = Normal(s_->face, s_->points);
    if (SameDirection(s_->points[s_->face.index[0]], n)) n = -n;
    return {s, n, false};
  } else if (N == 3) {
    const Simplex<3> *s_ = dynamic_cast<const Simplex<3> *>(s);
    if (s_->inside(Vector3f{0})) {
      return {{}, {}, true};
    } else {
      // traverse the faces of s
      std::pair<Face, float> result = std::make_pair(
          s_->faces[0], DistanceToFace(s_->faces[0], s_->points, Vector3f{0}));
      // Select the face with minimum distance
      for (int i = 1; i < 4; ++i) {
        float distance = DistanceToFace(s_->faces[i], s_->points, Vector3f{0});
        if (distance < result.second)
          result = std::make_pair(s_->faces[i], distance);
      }

      // Acquire the normal of the face
      Normal3f n = Normal(result.first, s_->points);
      if (SameDirection(s_->points[result.first.index[0]], n)) n = -n;
      return {new Simplex<2>(result.first, s_->points), n, false};
    }
  } else {
    TODO();
    return {};
  }
}
}  // namespace detail_
/**
 * @brief Implementation of GJK algorithm
 * @see
 * https://en.wikipedia.org/wiki/Gilbert%E2%80%93Johnson%E2%80%93Keerthi_distance_algorithm
 *
 * @param p The first ConvexHull to intersect
 * @param q The second ConvexHull to intersect
 * @return true iff there exists a intersection
 * @pseudo
  function GJK_intersection(shape p, shape q, vector initial_axis):
    vector  A = Support(p, initial_axis) − Support(q, −initial_axis)
    simplex s = {A}
    vector  D = −A

    loop:
        A = Support(p, D) − Support(q, −D)
        if dot(A, D) < 0:
            reject
        s = s ∪ A
        s, D, contains_origin := NearestSimplex(s)
        if contains_origin:
            accept
 */
inline bool GJKIntersection(const ConvexHullInstance &p,
                            const ConvexHullInstance &q) {
  using namespace detail_;
  Vector3f     initial_axis = Vector3f{1, 0, 0};
  Vector3f     A       = Support(p, initial_axis) - Support(q, -initial_axis);
  SimplexBase *s       = new Simplex<0>{A};
  Vector3f     D       = -A;
  bool         success = false;

  int max_iter = 128;
  while (true) {
    A = Support(p, D) - Support(q, -D);
    if (Dot(A, D) < 0) return false;
    auto s_ = s->promote(A);
    delete s;  // replacement
    s = s_;
    if (max_iter-- <= 0) break;
    std::tie(s_, D, success) = NearestSimplex(s);
    if (success) return true;
    delete s;  // replacement
    s = s_;
  }

#if 0
  fmt::print("{}\n", s->getSize());
  const auto s_ = dynamic_cast<Simplex<3> *>(s);
  fmt::print("{} {} {} {}\n", s_->points[0].toString(),
             s_->points[1].toString(), s_->points[2].toString(),
             s_->points[3].toString());
  for (int i = 0; i < 4; ++i) {
    assert(s_->faces[i].index[0] != s_->faces[i].index[1]);
    assert(s_->faces[i].index[1] != s_->faces[i].index[2]);
  }
#endif

  return true;
}
}  // namespace experimental
FLG_NAMESPACE_END

#endif
