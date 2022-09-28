#ifndef __RNG_HPP__
#define __RNG_HPP__

#include <random>
#include <thread>

#include "common/rng.h"
#include "common/vector.h"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

/**
 * Random generator for the render.
 * Mostly used by sampler.
 * @note This implementation is not thread safe, and the member functions should
 * be only called within a thread.
 */
class RandomCPU : public Random {
public:
  RandomCPU(uint32_t seed = 0) : m_seed(seed), m_generator(m_seed) {}
  Float get1D() {
    static std::uniform_real_distribution<Float> distribution(0, 1);
    return distribution(m_generator);
  }  // Float get1D()
  Vector2f get2D() { return {get1D(), get1D()}; }

private:
  uint32_t     m_seed;
  std::mt19937 m_generator;
};

FLG_NAMESPACE_END

#endif