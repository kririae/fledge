#include "bvh_test.hpp"

#include <iostream>
#include <limits>
#include <memory_resource>

#include "base_bvh.hpp"
#include "plymesh.hpp"
#include "rng.hpp"
#include "shape.hpp"

using namespace fledge;
using namespace fledge::experimental;
using namespace std::literals::chrono_literals;

static void test_BasicBVHBuilder() {
  BVHTester tester(EBVHType::EBVHRadix, std::pmr::get_default_resource());
  tester.loadSphere();
  tester.build();
  if (!tester.correctness1()) SErr("correctness test 1 failed");
}

int main() {
  test_BasicBVHBuilder();
}