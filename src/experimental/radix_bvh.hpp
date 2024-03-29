/**
 * @file radix_bvh.hpp
 * @author Zike (Kevin) XU (kririae@outlook.com)
 * @brief This file implements the classical parallel BVH construction based on
 * radix tree.
 * @version 0.1
 * @date 2022-10-16
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __RADIX_BVH_HPP__
#define __RADIX_BVH_HPP__

#include <math.h>

#include <cstdint>
#include <memory_resource>

#include "experimental/base_bvh.hpp"
#include "experimental/experimental_utils.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN
namespace experimental {
/**
 * @brief see
 * https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf
 * @note This implementation will duplicate the vertices
 */
class RadixBVHBuilder : public BVHBuilderBase {
public:
  // Leaf node
  struct RadixTriangle : public Triangle {
    uint32_t morton_code, parent;
  };

  // Internal node
  struct RadixBVHNode {
    BVHBound bound{};
    int      n_triangles{}, split_point{}, parent{};
    int      flag{};

    // Radix-related information
    int     direction{};             // int for now
    uint8_t left_node_type : 1 {};   // 0: internal; 1: leaf
    uint8_t right_node_type : 1 {};  // 0: internal; 1: leaf
  };

  RadixBVHBuilder(InternalTriangleMesh      *mesh,
                  std::pmr::memory_resource *mem_resource)
      : BVHBuilderBase(mesh, mem_resource),
        m_upstream(m_mem_resource),
        m_resource(&m_upstream) {}
  ~RadixBVHBuilder() override {}

  void     build() override;
  BVHBound getBound() const override { return m_internal_nodes[0].bound; }
  bool     intersect(BVHRayHit &rayhit) override;

protected:
  void prestage();
  void parallelBuilder();
  bool recursiveIntersect(RadixBVHNode *node, BVHRayHit &rayhit,
                          std::size_t &n_intersect);
  /* delta function in the paper */
  __always_inline int delta(int i, int j) {
    if (j < 0 || j >= m_n_triangles) return -1;
    if (m_triangles[i].morton_code == m_triangles[j].morton_code)
      return __builtin_clz(i ^ j) + 32;
    return __builtin_clz(m_triangles[i].morton_code ^
                         m_triangles[j].morton_code);
  }

  int            m_n_triangles;
  BVHBound       m_bound;
  RadixTriangle *m_triangles;
  RadixBVHNode  *m_internal_nodes;

private:
  tbb::cache_aligned_resource         m_upstream{m_mem_resource};
  std::pmr::monotonic_buffer_resource m_resource{&m_upstream};
};
}  // namespace experimental
FLG_NAMESPACE_END

#endif
