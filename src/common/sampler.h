#ifndef __SAMPLER_H__
#define __SAMPLER_H__

#include "debug.hpp"
#include "fledge.h"
#include "rng.hpp"

FLG_NAMESPACE_BEGIN

class Sampler {
public:
  Sampler(uint64_t SPP, uint32_t seed = 0) : m_SPP(SPP), m_rng(seed) {}
  void     setPixel(const Vector2f &p) { m_p = p; }
  Float    get1D() { return m_rng.get1D(); }
  Vector2f get2D() { return m_rng.get2D(); }
  bool     reset() { return true; }  // doing nothing in this trivial sampler
  Vector2f getPixelSample() {
    // Stay naive for now
    return m_rng.get2D() - Vector2f(0.5) + m_p;
  }

  uint64_t m_SPP;
  Vector2f m_p;
  Random   m_rng;
};

FLG_NAMESPACE_END

#endif
