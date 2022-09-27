#ifndef __RNG_H__
#define __RNG_H__

#include "common/switcher.h"
#include "common/vector.h"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

class RandomCPU;
class RandomGPU;

/**
 * Random generator for the render.
 * Mostly used by sampler.
 * @note This implementation is not thread safe, and the member functions should
 * be only called within a thread.
 */
class Random : public Switcher<RandomCPU, RandomGPU> {
public:
  using Switcher::Switcher;
  virtual ~Random() = default;
  F_CPU_GPU Float    get1D();
  F_CPU_GPU Vector2f get2D();

  virtual Float get1D_impl() {
    assert(false);
    return 0.0;
  }

  virtual Vector2f get2D_impl() {
    assert(false);
    return Vector2f{0.0};
  }
};

FLG_NAMESPACE_END

#endif
