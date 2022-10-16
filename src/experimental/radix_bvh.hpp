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
#include <oneapi/tbb/detail/_machine.h>

#include <memory_resource>

#include "experimental/base_bvh.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN
namespace experimental {
namespace detail_ {
uint32_t Next2Pow3k(uint32_t n) {
  // In general, we'll split the larges BBox into 2^k * 2^k * 2^k lattices.
  // So k should be acquired in order to assign the z-order curve
  double n_  = static_cast<double>(n);
  n_         = pow(n_, 1.0 / 3);
  n_         = log2(n_);
  uint32_t k = static_cast<uint32_t>(ceil(n_));
  assert(pow(2, 3 * k) >= n);
  return pow(2, 3 * k);
}

uint32_t Next2Pow(uint32_t n) {
  double   n_ = log2(static_cast<double>(n));
  uint32_t k  = static_cast<uint32_t>(ceil(n_));
  return pow(2, k);
}

// From
// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
__always_inline uint32_t ExpandBits(uint32_t v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__always_inline uint32_t MortonCurve3D(Vector3f normalized_coordinate) {
  // assume that there's at most 2^30 primitives(triangles in our case)
  constexpr uint32_t factor = 1024;
  uint32_t           x =
      std::clamp<uint32_t>(normalized_coordinate.x() * factor, 0, factor - 1);
  uint32_t y =
      std::clamp<uint32_t>(normalized_coordinate.y() * factor, 0, factor - 1);
  uint32_t z =
      std::clamp<uint32_t>(normalized_coordinate.z() * factor, 0, factor - 1);
  x = ExpandBits(x);
  y = ExpandBits(y);
  z = ExpandBits(z);
  return (x << 2) + (y << 1) + z;
}
}  // namespace detail_
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
    BVHBound       bound;
    RadixTriangle *triangles;
    int            n_triangles, split_point, parent;
    int            flag;

    // Radix-related information
    int     direction;            // int for now
    uint8_t left_node_type : 1;   // 0: internal; 1: leaf
    uint8_t right_node_type : 1;  // 0: internal; 1: leaf
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
  bool recursiveIntersect(RadixBVHNode *node, BVHRayHit &rayhit);
  /* delta function in the paper */
  __always_inline int delta(int i, int j) {
    if (j < 0 || j >= m_n_triangles) return -1;
    uint64_t m1 = ((uint64_t)m_triangles[i].morton_code << 32) | i;
    uint64_t m2 = ((uint64_t)m_triangles[j].morton_code << 32) | j;
    return __builtin_clz(m1 ^ m2);
  }

  int            m_n_triangles;
  BVHBound       m_bound;
  RadixTriangle *m_triangles;
  RadixBVHNode  *m_internal_nodes;

private:
  tbb::cache_aligned_resource          m_upstream{m_mem_resource};
  std::pmr::synchronized_pool_resource m_resource{&m_upstream};
};
}  // namespace experimental
FLG_NAMESPACE_END

#endif
