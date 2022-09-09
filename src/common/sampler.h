#ifndef __SAMPLER_H__
#define __SAMPLER_H__

#include <cstdint>
#include <limits>

#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"
#include "rng.hpp"

FLG_NAMESPACE_BEGIN

class Sampler {
public:
  Sampler(uint64_t SPP, uint32_t seed) : m_SPP(SPP), m_rng(seed) {}
  virtual void     setPixel(const Vector2f &p) { m_p = p; }
  virtual Float    get1D() { return m_rng.get1D(); }
  virtual Vector2f get2D() { return m_rng.get2D(); }
  virtual bool     reset() {
        return true;
  }  // doing nothing in this trivial sampler
  virtual Vector2f getPixelSample() {
    // Stay naive for now
    return m_rng.get2D() - Vector2f(0.5) + m_p;
  }

protected:
  uint64_t m_SPP;
  Vector2f m_p;
  Random   m_rng;
};

namespace detail_ {
inline int *GetPrimeList() {
  constexpr static int N = 16384;
  static int           prime[N]{};
  if (prime[0] == 0) {
    // Linear Sieve
    bool m_flag[N]{};
    int  cnt = 0;
    for (int i = 2; i < N; ++i) {
      if (!m_flag[i]) prime[cnt++] = i;
      for (int j = 0; j < cnt; ++j) {
        int v_ = i * prime[j];  // TODO
        if (v_ >= N) break;
        m_flag[v_] = 1;
        if (i % prime[j] == 0) break;
      }
    }
  }  // if(prime[0] == 0)
  return &prime[0];
}

inline Float RadicalInverse_(uint64_t a, int base) {
  Float    inv_base        = (Float)1 / (Float)base;
  uint64_t reversed_digits = 0;
  Float    inv_base_n      = 1;
  while (a) {
    uint64_t next   = a / base;
    uint64_t digit  = a - next * base;
    reversed_digits = reversed_digits * base + digit;
    inv_base_n *= inv_base;
    a = next;
  }
  return reversed_digits * inv_base_n;
}
}  // namespace detail_

/**
 * @brief Given the parameter a, return the radical inverse of a, using the
 * index'th prime as the base
 *
 * @param a
 * @param index The index of the prime number
 * @return Float
 */
inline Float RadicalInverse(uint64_t a, int index) {
  return detail_::RadicalInverse_(a, detail_::GetPrimeList()[index]);
}

/**
 * HaltonSampler will generate samples through Halton Sequence globally.
 */
class HaltonSampler : public Sampler {
public:
  HaltonSampler(uint64_t SPP, Vector2d sample_interval)
      : Sampler(SPP, 0), m_cnt(0) {
    for (int i = 0; i < 2; ++i) {
      int base  = i == 0 ? 2 : 3;
      int scale = 1, exp = 0;
      // Find the minimum 2^i or 3^j that is larger than any of the ...
      while (scale < std::min(sample_interval[i], M_MAX_RESOLUTION)) {
        scale *= base;
        ++exp;
      }  // scale < ...
      m_base_scales[i] = scale;
      m_base_exp[i]    = exp;
    }  // i < 2
    m_sample_stride = m_base_scales[0] * m_base_scales[1];
  }
  void setPixel(const Vector2f &p) override {
    Sampler::setPixel(p);
    m_cnt = 0;
  }
  Float get1D() override {
    Float ret = 0;
    return ret;
  }
  Vector2f get2D() override { return {get1D(), get1D()}; }
  bool     reset() override {
    m_cnt = 0;
    return true;
  }  // doing nothing in this trivial sampler
  Vector2f getPixelSample() override {
    // Stay naive for now
    TODO();
  }

protected:
  uint64_t             m_cnt, m_sample_stride;
  Vector2d             m_base_scales, m_base_exp;
  static constexpr int M_MAX_RESOLUTION = 128;
  /**
   * @brief Inverse mapping from the current pixel and the given sample index to
   * global index into the overall set of sample vectors. Adapted From PBRT's
   * interface.
   *
   * @param sample_num
   * @return uint64_t
   */
  virtual uint64_t getIndexForSample(uint64_t sample_num) { TODO(); }
  Vector2d         m_pixel_for_offset;
  uint64_t         m_offset_for_pixel;
};

FLG_NAMESPACE_END

#endif
