#ifndef __SAMPLER_H__
#define __SAMPLER_H__

#include <cstdint>
#include <limits>

#include "common/math_utils.h"
#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"
#include "rng.hpp"

FLG_NAMESPACE_BEGIN

static constexpr float ONE_MINUS_EPSILON = 0.99999994;

class Sampler {
public:
  Sampler(uint64_t SPP, uint32_t seed) : m_SPP(SPP), m_rng(seed) {}
  virtual ~Sampler() = default;
  virtual void     setPixel(const Vector2d &p) { m_p = p; }
  virtual Float    get1D() { return m_rng.get1D(); }
  virtual Vector2f get2D() { return m_rng.get2D(); }
  virtual bool     reset() {
        return true;
  }  // doing nothing in this trivial sampler
  virtual Vector2f getPixelSample() {
    // Stay naive for now
    return m_rng.get2D() + m_p.cast<Float, 2>();
  }

protected:
  uint64_t m_SPP;
  Vector2d m_p;
  Random   m_rng;
};

namespace detail_ {
inline int *GetPrimeList() {
  constexpr static int N = 16384 * 16;
  static int           prime[N]{};
  if (prime[0] == 0) {
    // The common Linear Sieve
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

inline Float RadicalInverse_(int base, uint64_t a) {
  assert(base > 0);
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

// https://oi-wiki.org/math/number-theory/gcd/
// and https://oi-wiki.org/math/number-theory/inverse/
// ax + by = \gcd(a, b), given a and b, solve x, y
inline void ExGCD(uint64_t a, uint64_t b, int64_t &x, int64_t &y) {
  if (b == 0) {
    x = 1, y = 0;
    return;
  }  // if b == 0
  ExGCD(b, a % b, y, x);
  y -= a / b * x;
}

/**
 * @brief Modular Multiplicative Inverse
 * @note $ax \equiv 1 \pmod{b}$, which leads to ax + kb = 1
 *
 * @return uint64_t x
 */
inline uint64_t MulModInverse(int64_t a, int64_t b) {
  int64_t x, y;
  ExGCD(a, b, x, y);
  return Mod(x, b);
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
inline Float RadicalInverse(int base_index, uint64_t a) {
  return detail_::RadicalInverse_(detail_::GetPrimeList()[base_index], a);
}

template <int base>
inline uint64_t InverseRadicalInverse(uint64_t inverse, int n_digits) {
  uint64_t index = 0;
  for (int i = 0; i < n_digits; ++i) {
    uint64_t digit = inverse % base;
    inverse /= base;
    index = index * base + digit;
  }
  return index;
}

// clang-format off
/**
 * HaltonSampler will generate samples through Halton Sequence globally.
 * The implementation is adapted from https://pbr-book.org/3ed-2018/Sampling_and_Reconstruction/The_Halton_Sampler

                                         ────────────────────────►
                        SAMPLE_INDEX        Get1D() or rayDepth
 ▲ HALTON_SEQ    ┌──  ┌───────────────┬────────────────────────────────────
──────────────►  │    │       1       │    f_2(a)   f_3(a)   f_5(a)   ...      for pixel (0, 0)
                 │    ├───────────────┼────────────────────────────────────
                 │    │       2       │    f_2(a+1) f_3(a+1) f_5(a+1) ...           ...  │
                 │    ├───────────────┼────────────────────────────────────              │
         SPP 1   │    │       3       │    f_2(a+2) f_3(a+2) ...                   SPP++ │
                 .    ├───────────────┼────────────────────────────────────              │
                 .    │       4       │    f_2(a+3) ...                                  │
                 .    └───────────────┼────────────────────────────────────              ▼
                 │           ...      │
                 └──                  │                                        for pixel (0, 0)
         SPP 2: ...
  */
// clang-format on
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
    m_mul_inverse[0] =
        detail_::MulModInverse(m_base_scales[1], m_base_scales[0]);
    m_mul_inverse[1] =
        detail_::MulModInverse(m_base_scales[0], m_base_scales[1]);
  }
  ~HaltonSampler() override = default;
  void setPixel(const Vector2d &p) override {
    Sampler::setPixel(p);
    m_cnt                = 0;
    m_pixel_sample_index = 0;
    m_sample_index       = getIndexForSample(0);
  }
  Float get1D() override {
    return std::min(sampleDimension(m_sample_index, m_cnt++),
                    ONE_MINUS_EPSILON);  // avoid numerical issues
  }
  Vector2f get2D() override { return {get1D(), get1D()}; }
  bool     reset() override {
    m_cnt = 0;
    // increase the pixel sample index and find the index in global sample
    // vector
    m_sample_index = getIndexForSample(m_pixel_sample_index + 1);
    return ++m_pixel_sample_index < m_SPP;
  }  // doing nothing in this trivial sampler
  Vector2f getPixelSample() override {
    // Stay naive for now
    return m_p.cast<Float, 2>() + get2D();
  }

protected:
  uint64_t             m_cnt, m_sample_stride;
  uint64_t             m_pixel_sample_index{0}; /* the value that < SPP */
  uint64_t             m_sample_index{0};       /* the value that < SPP */
  Vector2d             m_base_scales, m_base_exp;
  Vector2d             m_pixel_for_offset;
  Vector2d             m_mul_inverse;
  uint64_t             m_offset_for_pixel;
  static constexpr int M_MAX_RESOLUTION = 128;

  /**
   * @brief Inverse mapping from the current pixel and the given sample index to
   * global index into the overall set of sample vectors. Adapted From PBRT's
   * interface.
   *
   * @param sample_num
   * @return uint64_t
   */
  virtual uint64_t getIndexForSample(uint64_t sample_num) {
    if (m_p != m_pixel_for_offset) {
      // Compute Halton sample offset for _currentPixel_
      m_offset_for_pixel = 0;
      if (m_sample_stride > 1) {
        Vector2d pm(Mod(m_p[0], M_MAX_RESOLUTION),
                    Mod(m_p[1], M_MAX_RESOLUTION));
        for (int i = 0; i < 2; ++i) {
          uint64_t dim_offset =
              (i == 0) ? InverseRadicalInverse<2>(pm[i], m_base_exp[i])
                       : InverseRadicalInverse<3>(pm[i], m_base_exp[i]);
          m_offset_for_pixel += dim_offset *
                                (m_sample_stride / m_base_scales[i]) *
                                m_mul_inverse[i];
        }
        m_offset_for_pixel %= m_sample_stride;
      }  // if
      m_pixel_for_offset = m_p;
    }

    return m_offset_for_pixel + sample_num * m_sample_stride;
  }
  /**
   * @brief The core function relates (SPP, Pixel, Dimension) to *Halton
   * Sequence*.
   * @note The correlation is, first, Halton Sequence can be indexed with
   * two params, THE index of the prime number and *a*. When Dimension
   * increased, Halton Sequence goes in the direction that a remains the same.
   * @see The figure in HaltonSampler class
   *
   * @param index *intervalSampleIndex* in PBRT's context, which is obtained
   * from the number of SPP.
   * @param cnt
   * @return Float
   */
  virtual Float sampleDimension(uint64_t index, int cnt) {
    // So the variable within a k
    switch (cnt) {
      case 0:
        return RadicalInverse(cnt, index / m_base_scales[0]);
      case 1:
        return RadicalInverse(cnt, index / m_base_scales[1]);
      default:
        return RadicalInverse(cnt, index);
    }
  }
};

FLG_NAMESPACE_END

#endif
