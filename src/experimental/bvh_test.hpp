#ifndef __EXPERIMENTAL_BVH_TEST_HPP__
#define __EXPERIMENTAL_BVH_TEST_HPP__

#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/core.h>

#include <chrono>

#include "fledge.h"

FLG_NAMESPACE_BEGIN
namespace experimental {
struct Event {
  using clock = std::chrono::high_resolution_clock;
  Event(const std::string &desc) : m_start(clock::now()), m_desc(desc) {}
  void end() {
    using namespace std::literals::chrono_literals;
    fmt::print(fg(fmt::color::steel_blue), "[{}] takes {} ms\n", m_desc,
               (clock::now() - m_start) / 1ms);
  }

private:
  decltype(clock::now()) m_start;
  std::string            m_desc;
};
}  // namespace experimental
FLG_NAMESPACE_END

#endif
