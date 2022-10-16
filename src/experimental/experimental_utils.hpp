#ifndef __EXPERIMENTAL_UTILS_HPP__
#define __EXPERIMENTAL_UTILS_HPP__

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <functional>

#include "fledge.h"

FLG_NAMESPACE_BEGIN
namespace experimental {
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
