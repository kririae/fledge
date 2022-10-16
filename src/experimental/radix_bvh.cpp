#include "experimental/radix_bvh.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/parallel_sort.h>

#include <execution>
#include <memory_resource>
#include <vector>

#include "experimental/experimental_utils.hpp"
#include "experimental/intersector.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN
namespace experimental {

void RadixBVHBuilder::build() {
  prestage();
  parallelBuilder();
  assert(m_internal_nodes->bound.lower == m_bound.lower);
  assert(m_internal_nodes->bound.upper == m_bound.upper);
  assert(m_internal_nodes->n_triangles == m_n_triangles);
}

bool RadixBVHBuilder::intersect(BVHRayHit &rayhit) {
  std::size_t n_intersect = 0;
  bool res = recursiveIntersect(&m_internal_nodes[0], rayhit, n_intersect);
  return res;
}

void RadixBVHBuilder::prestage() {
  m_n_triangles = m_mesh->nInd / 3;
  m_triangles =
      std::pmr::polymorphic_allocator<RadixTriangle>{&m_resource}.allocate(
          m_n_triangles);
  // be sure to initialize
  m_internal_nodes =
      std::pmr::polymorphic_allocator<RadixBVHNode>{&m_resource}.allocate(
          m_n_triangles - 1);

  ParallelForLinear(0, m_n_triangles, [&](std::size_t i) {
    m_triangles[i].m_mesh = m_mesh;
    m_triangles[i].m_v    = m_mesh->ind + i * 3;
  });

  // First pass, calculate the global bound
  // TODO: encapsulate the reduce function
  m_bound = tbb::parallel_reduce(
      tbb::blocked_range<std::size_t>(0, m_n_triangles), BVHBound{},
      [&](const tbb::blocked_range<std::size_t> &r, BVHBound init) -> BVHBound {
        for (std::size_t i = r.begin(); i != r.end(); ++i)
          init.merge(m_triangles[i].getBound());
        return init;
      },
      [](const BVHBound &b1, const BVHBound &b2) -> BVHBound {
        BVHBound bound = b1;
        bound.merge(b2);
        return bound;
      });

  // TODO: make sure there's no false sharing
  Vector3f edges = m_bound.upper - m_bound.lower;
  ParallelForLinear(0, m_n_triangles, [&](std::size_t i) {
    const auto bound                 = m_triangles[i].getBound();
    Vector3f   centroid              = (bound.lower + bound.upper) / 2;
    Vector3f   normalized_coordinate = (centroid - m_bound.lower) / edges;
    assert(0 <= normalized_coordinate.x() && normalized_coordinate.x() <= 1.0);
    assert(0 <= normalized_coordinate.y() && normalized_coordinate.y() <= 1.0);
    assert(0 <= normalized_coordinate.z() && normalized_coordinate.z() <= 1.0);
    m_triangles[i].morton_code =
        detail_::EncodeMorton3(normalized_coordinate * (1 << 10));
  });

  // TODO: replace this implementation with
  // https://github.com/google/highway/tree/master/hwy/contrib/sort
  tbb::parallel_sort(
      m_triangles, m_triangles + m_n_triangles,
      [](const RadixTriangle &a, const RadixTriangle &b) -> bool {
        return a.morton_code < b.morton_code;
      });

#if 0  // Vertex sorting
  struct MortonVertex {
    Vector3f position;
    int      original_index;
    uint32_t morton_code;
  };
  std::pmr::vector<MortonVertex> vertices(m_mesh->nVert, m_mem_resource);
  std::pmr::vector<int>          vertex_map(m_mesh->nVert, m_mem_resource);
  ParallelForLinear(0, m_mesh->nVert, [&](std::size_t i) {
    vertices[i] =
        MortonVertex{.position       = m_mesh->p[i],
                     .original_index = static_cast<int>(i),
                     .morton_code    = detail_::EncodeMorton3(
                            (1 << 10) * (m_mesh->p[i] - m_bound.lower) / edges)};
  });
  tbb::parallel_sort(vertices.begin(), vertices.end(),
                     [](const MortonVertex &a, const MortonVertex &b) -> bool {
                       return a.morton_code < b.morton_code;
                     });
  ParallelForLinear(0, m_mesh->nVert, [&](std::size_t i) {
    vertex_map[vertices[i].original_index] = i;
  });
  // Update all indices
  ParallelForLinear(0, m_mesh->nInd, [&](std::size_t i) {
    m_mesh->ind[i] = vertex_map[m_mesh->ind[i]];
  });
  m_mesh->p = std::pmr::polymorphic_allocator<Vector3f>{&m_resource}.allocate(
      m_mesh->nVert);
  ParallelForLinear(0, m_mesh->nVert, [&](std::size_t i) {
    m_mesh->p[i] = vertices[i].position;
  });
#endif
}

void RadixBVHBuilder::parallelBuilder() {
  ParallelForLinear(0, m_n_triangles - 1, [&](std::size_t i) {
    RadixBVHNode &internal_node = m_internal_nodes[i];
    internal_node.direction     = Sign(delta(i, i + 1) - delta(i, i - 1));
    const int d                 = internal_node.direction;
    assert(d != 0);

    // move toward the sibling node
    int delta_min = delta(i, i - d);
    int l_max     = 2;
    while (delta(i, i + l_max * d) > delta_min) /* obtain the maximum l_max */
      l_max <<= 1;

    // Moving rightward
    int l = 0;
    for (int t = l_max / 2; t >= 1; t >>= 1)
      if (delta(i, i + (l + t) * d) > delta_min) l += t;
    int j = i + l * d;  // the other bound, inclusive

    int delta_node = delta(i, j);
    int s          = 0;
    for (int t = detail_::Next2Pow(l); t >= 1; t >>= 1)
      if (delta(i, i + (s + t) * d) > delta_node) s += t;

    // Finally decide the split point, always chose the left one, neglecting the
    // internal_node.direction
    internal_node.split_point = i + s * d + std::min(d, 0);
    const auto split_point    = internal_node.split_point;
    if (std::min(static_cast<int>(i), j) == split_point)
      internal_node.left_node_type = 1;
    if (std::max(static_cast<int>(i), j) == split_point + 1)
      internal_node.right_node_type = 1;
    internal_node.n_triangles = j - i + 1;

    // Final pass, calculate the parent
    if (internal_node.left_node_type)
      m_triangles[split_point].parent = i;
    else
      m_internal_nodes[split_point].parent = i;

    if (internal_node.right_node_type)
      m_triangles[split_point + 1].parent = i;
    else
      m_internal_nodes[split_point + 1].parent = i;
  });

  ParallelForLinear(0, m_n_triangles, [&](std::size_t i) {
    int      current_index  = m_triangles[i].parent;
    int      previous_index = i;
    BVHBound subnode_bound  = m_triangles[i].getBound();
    assert(0 <= current_index && current_index < m_n_triangles - 1);
    while (true) {
      auto &internal_node = m_internal_nodes[current_index];
      // Increment the flag by one, and fetch the original value
      int previous_flag = __sync_fetch_and_add(&internal_node.flag, 1);
      if (previous_flag == 0)
        break;
      else if (previous_flag == 1) {
        BVHBound other_bound{};
        int      other_num = 0;
        if (previous_index == internal_node.split_point) {
          other_bound =
              internal_node.right_node_type == 1
                  ? m_triangles[internal_node.split_point + 1].getBound()
                  : m_internal_nodes[internal_node.split_point + 1].bound;
        } else {
          assert(previous_index == internal_node.split_point + 1);
          other_bound = internal_node.left_node_type == 1
                          ? m_triangles[internal_node.split_point].getBound()
                          : m_internal_nodes[internal_node.split_point].bound;
        }

        internal_node.bound = BVHBound{};
        internal_node.bound.merge(other_bound);
        internal_node.bound.merge(subnode_bound);
      }

      subnode_bound = internal_node.bound;
      if (current_index == 0 && m_internal_nodes[0].parent == 0) break;
      previous_index = current_index;
      current_index  = m_internal_nodes[current_index].parent;
    }
  });
}

bool RadixBVHBuilder::recursiveIntersect(RadixBVHNode *node, BVHRayHit &rayhit,
                                         std::size_t &n_intersect) {
  float tnear, tfar;
  bool  inter = node->bound.intersect(rayhit.ray_o, rayhit.ray_d, tnear, tfar);
  if (!inter) return false;
  if (rayhit.tfar < tnear) return false;

  int      n_triangles = 0;
  Triangle triangles[2];
  if (node->left_node_type)
    triangles[n_triangles++] = m_triangles[node->split_point];
  if (node->right_node_type)
    triangles[n_triangles++] = m_triangles[node->split_point + 1];

  bool hit = false;
  tfar     = rayhit.tfar;
  // leaf node intersection
  for (int i = 0; i < n_triangles; ++i) {
    Triangle &triangle = triangles[i];
    float     thit;
    Vector3f  ng;
    ++n_intersect;
    bool inter = detail_::PlueckerTriangleIntersect(triangle.a(), triangle.b(),
                                                    triangle.c(), rayhit.ray_o,
                                                    rayhit.ray_d, &thit, &ng);
    if (!inter) continue;
    if (thit > tfar) continue;
    hit           = true;
    tfar          = thit;
    rayhit.hit    = true;
    rayhit.tfar   = tfar;
    rayhit.hit_ng = rayhit.hit_ns = ng;
  }

  bool res_left = false, res_right = false;
  if (!node->left_node_type)
    res_left = recursiveIntersect(&m_internal_nodes[node->split_point], rayhit,
                                  n_intersect);
  if (!node->right_node_type)
    res_right = recursiveIntersect(&m_internal_nodes[node->split_point + 1],
                                   rayhit, n_intersect);
  if (res_left || res_right || hit) assert(rayhit.hit == true);
  return res_left || res_right || hit;
}

}  // namespace experimental
FLG_NAMESPACE_END
