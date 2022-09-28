#include "common/rng.h"

#include "rng.hpp"

FLG_NAMESPACE_BEGIN

F_CPU_GPU Float Random::get1D() {
  auto invoker = [&](auto cls) { return cls->get1D(); };
  return dispatch(invoker);
}

F_CPU_GPU Vector2f Random::get2D() {
  auto invoker = [&](auto cls) { return cls->get2D(); };
  return dispatch(invoker);
}

FLG_NAMESPACE_END
