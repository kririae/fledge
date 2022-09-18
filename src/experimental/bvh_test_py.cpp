#include <nanobind/nanobind.h>

#include "bvh_test.hpp"
#include "fledge.h"

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(bvh_test_py, m) {
  using namespace fledge;
  using namespace fledge::experimental;
  // nb::class_<BVHTester>
}