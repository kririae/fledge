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

__always_inline uint32_t LeftShift3(uint32_t x) {
  if (x == (1 << 10)) --x;
  x = (x | (x << 16)) & 0b00000011000000000000000011111111;
  x = (x | (x << 8)) & 0b00000011000000001111000000001111;
  x = (x | (x << 4)) & 0b00000011000011000011000011000011;
  x = (x | (x << 2)) & 0b00001001001001001001001001001001;
  return x;
}

inline uint32_t EncodeMorton3(const Vector3f &v) {
  return (LeftShift3(v.z()) << 2) | (LeftShift3(v.y()) << 1) |
         LeftShift3(v.x());
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
