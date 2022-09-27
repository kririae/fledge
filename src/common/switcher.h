#ifndef __SWITCHER_HPP__
#define __SWITCHER_HPP__

#include <type_traits>

#include "fledge.h"

FLG_NAMESPACE_BEGIN

F_CPU_GPU constexpr bool is_on_gpu() {
#ifdef __CUDA_ARCH__
  return true;
#else
  return false;
#endif
}

/**
 * @brief Switcher is the crucial class in coordinating CPU and GPU
 * implementation. Tc accept a class that can be executed on both CPU, and Tg
 * accept the one that can only be executed on GPU.
 * @note the dispatch is static, so don't worry about the performance at all,
 * the only overhead is that is takes two sizeof(pointer).
 *
 * @tparam Tc
 * @tparam Tg
 */
template <typename Tc, typename Tg>
class Switcher {
public:
  using ptr_type = void *;

  Switcher() = default;  // retain the default ctor
  F_CPU_GPU Switcher(Tc *ptr_cpu, Tg *ptr_gpu)
      : ptr_cpu(ptr_cpu), ptr_gpu(ptr_gpu) {}
  // Switcher can be trivially copied

  template <typename T = void>
  F_CPU_GPU T *ptr() {
    if constexpr (is_on_gpu()) {
      return ptr_gpu;
    } else {
      return ptr_cpu;
    }
  }

  template <typename F>
  F_CPU_GPU decltype(auto) dispatch(F &&func) {
    if constexpr (is_on_gpu()) {
      return func(reinterpret_cast<Tg *>(ptr_gpu));
    } else {
      return func(reinterpret_cast<Tc *>(ptr_cpu));
    }
  }

  template <typename F>
  F_CPU_GPU decltype(auto) dispatch(F &&func) const {
    if constexpr (is_on_gpu()) {
      return func(reinterpret_cast<const Tg *>(ptr_gpu));
    } else {
      return func(reinterpret_cast<const Tc *>(ptr_cpu));
    }
  }

private:
  ptr_type ptr_cpu, ptr_gpu;
};

FLG_NAMESPACE_END
#endif
