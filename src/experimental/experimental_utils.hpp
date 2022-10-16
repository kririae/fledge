#ifndef __EXPERIMENTAL_UTILS_HPP__
#define __EXPERIMENTAL_UTILS_HPP__

#include <math.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <cstdint>
#include <functional>

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

void ParallelForLinear(std::size_t begin, std::size_t end,
                       std::function<void(std::size_t index)> func) {
  if constexpr (true) {
    tbb::parallel_for(tbb::blocked_range<std::size_t>(begin, end),
                      [=](const tbb::blocked_range<std::size_t> &r) -> void {
                        for (std::size_t i = r.begin(); i != r.end(); ++i)
                          func(i);
                      });
  } else {
#pragma omp parallel for
    for (std::size_t i = begin; i != end; ++i) func(i);
  }
}
}  // namespace experimental
FLG_NAMESPACE_END

#endif
