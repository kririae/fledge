#include "experimental/radix_bvh.hpp"

#include <execution>
#include <memory_resource>
#include <vector>

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
  m_internal_nodes =
      std::pmr::polymorphic_allocator<RadixBVHNode>{&m_resource}.allocate(
          m_n_triangles - 1);

  for (int i = 0; i < m_n_triangles; ++i) {
    m_triangles[i].m_mesh = m_mesh;
    m_triangles[i].m_v    = m_mesh->ind + i * 3;
  }

  // First pass, calculate the global bound
  // TODO: use reduce
  for (int i = 0; i < m_n_triangles; ++i) {
    const BVHBound bound = m_triangles[i].getBound();
    m_bound.merge(bound);
  }

  // TODO: make sure there's no false sharing
  for (int i = 0; i < m_n_triangles; ++i) {
    const auto bound    = m_triangles[i].getBound();
    Vector3f   centroid = (bound.lower + bound.upper) / 2;
    Vector3f   normalized_coordinate =
        (centroid - m_bound.lower) / (m_bound.upper - m_bound.lower);
    assert(0 <= normalized_coordinate.x() && normalized_coordinate.x() <= 1.0);
    assert(0 <= normalized_coordinate.y() && normalized_coordinate.y() <= 1.0);
    assert(0 <= normalized_coordinate.z() && normalized_coordinate.z() <= 1.0);
    // m_triangles[i].morton_code =
    // detail_::MortonCurve3D(normalized_coordinate);
    m_triangles[i].morton_code =
        detail_::EncodeMorton3(normalized_coordinate * (1 << 10));
  }

  // TODO: replace this implementation with
  // https://github.com/google/highway/tree/master/hwy/contrib/sort
  std::sort(std::execution::par, m_triangles, m_triangles + m_n_triangles,
            [](const RadixTriangle &a, const RadixTriangle &b) -> bool {
              return a.morton_code < b.morton_code;
            });
}

void        RadixBVHBuilder::parallelBuilder() {
#pragma omp parallel for schedule(static)
  // in parallel
  for (int i = 0; i < m_n_triangles - 1; ++i) {
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
    if (std::min(i, j) == split_point) internal_node.left_node_type = 1;
    if (std::max(i, j) == split_point + 1) internal_node.right_node_type = 1;
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
  }

#if 0
	for (int i = 0; i < m_n_triangles; ++i) {
		auto internal_node = m_internal_nodes[m_triangles[i].parent];
		if (!(i == internal_node.split_point ||
					i == internal_node.split_point + 1)) {
			fmt::print("i: {}, parent: {}, split_point: {}\n", i,
								 m_triangles[i].parent, internal_node.split_point);
			fmt::print("{} {} {}\n", m_triangles[i - 1].morton_code,
								 m_triangles[i].morton_code, m_triangles[i + 1].morton_code);
			fmt::print("{} {} {}\n", m_triangles[i - 1].parent, m_triangles[i].parent,
								 m_triangles[i + 1].parent);
		}
		assert(i == internal_node.split_point ||
					 i == internal_node.split_point + 1);
	}
#endif

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < m_n_triangles; ++i) {
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

#if 0
        if (internal_node.left_node_type && internal_node.right_node_type) {
          const RadixTriangle t1 = m_triangles[internal_node.split_point];
          const RadixTriangle t2 = m_triangles[internal_node.split_point + 1];
          const Vector3f      c1 = t1.center();
          const Vector3f      c2 = t2.center();
          BVHBound            bound = t1.getBound();
          bound.merge(t2.getBound());
          assert(bound.lower == internal_node.bound.lower);
          assert(bound.upper == internal_node.bound.upper);
        }
#endif
      }

      subnode_bound = internal_node.bound;
      if (current_index == 0 && m_internal_nodes[0].parent == 0) break;
      previous_index = current_index;
      current_index  = m_internal_nodes[current_index].parent;
    }
  }

#if 0
  fmt::print("{}\n", m_internal_nodes[0].bound.lower.toString(),
             m_internal_nodes[0].bound.upper.toString());
  fmt::print(
      "{} {}\n",
      m_internal_nodes[m_internal_nodes->split_point].bound.lower.toString(),
      m_internal_nodes[m_internal_nodes->split_point].bound.upper.toString());
  fmt::print("{} {}\n",
             m_internal_nodes[m_internal_nodes->split_point + 1]
                 .bound.lower.toString(),
             m_internal_nodes[m_internal_nodes->split_point + 2]
                 .bound.upper.toString());
#endif
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
