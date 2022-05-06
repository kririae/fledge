#ifndef __RNG_HPP__
#define __RNG_HPP__

#include <random>

#include "fwd.hpp"

SV_NAMESPACE_BEGIN

class Random {
public:
  Random() = default;
  // generate Float range from [0, 1]
  Float get1D() {
    static thread_local std::mt19937      generator{std::random_device{}()};
    std::uniform_real_distribution<Float> distribution(0, 1);
    return distribution(generator);
  }
  Vector2f get2D() { return {get1D(), get1D()}; }
};

SV_NAMESPACE_END

#endif
