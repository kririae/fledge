#ifndef __RNG_HPP__
#define __RNG_HPP__

#include <random>

#include "fwd.hpp"
#include "common/vector.h"

FLG_NAMESPACE_BEGIN

class Random {
public:
  Random(uint32_t seed = 0) : m_seed(seed) {}
  // generate Float range from [0, 1]
  Float get1D() {
    static thread_local std::mt19937      generator{m_seed};
    std::uniform_real_distribution<Float> distribution(0, 1);
    return distribution(generator);
  }
  Vector2f get2D() { return {get1D(), get1D()}; }

private:
  uint32_t m_seed;
};

FLG_NAMESPACE_END

#endif
