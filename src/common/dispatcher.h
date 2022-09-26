#ifndef __DISPATCHER_H__
#define __DISPATCHER_H__

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "fledge.h"

FLG_NAMESPACE_BEGIN

/**
 * This part is adapted from PBRT-v4 taggedptr.h
 */

template <typename... Ts>
struct TypePack {
  static constexpr std::size_t size = sizeof...(Ts);
};

template <typename T, typename Tp>
struct IndexOf {
  static constexpr std::size_t value = 0;
  static_assert(!std::is_void_v<T>, "Type not present in TypePack");
};

template <typename T, typename... Ts>
struct IndexOf<T, TypePack<T, Ts...>> {
  static constexpr std::size_t value = 0;
};

template <typename T, typename U, typename... Ts>
struct IndexOf<T, TypePack<U, Ts...>> {
  static constexpr std::size_t value = 1 + IndexOf<T, TypePack<Ts...>>::value;
};

template <typename T, typename Tp>
struct HasType {
  static constexpr bool value = false;
};

template <typename T, typename U, typename... Ts>
struct HasType<T, TypePack<U, Ts...>> {
  static constexpr bool value =
      (std::is_same_v<T, U> || HasType<T, TypePack<Ts...>>::value);
};

template <typename... Ts>
struct IsSameType;

template <>
struct IsSameType<> {
  static constexpr bool value = true;
};
template <typename T>
struct IsSameType<T> {
  static constexpr bool value = true;
};

template <typename T, typename U, typename... Ts>
struct IsSameType<T, U, Ts...> {
  static constexpr bool value =
      (std::is_same_v<T, U> && IsSameType<U, Ts...>::value);
};

template <typename... Ts>
struct SameType;

template <typename T, typename... Ts>
struct SameType<T, Ts...> {
  using type = T;
  static_assert(IsSameType<T, Ts...>::value,
                "Not all types in pack are the same");
};

template <typename F, typename... Ts>
struct ReturnType {
  using type =
      typename SameType<typename std::invoke_result_t<F, Ts *>...>::type;
};

template <typename F, typename... Ts>
struct ReturnTypeConst {
  using type =
      typename SameType<typename std::invoke_result_t<F, const Ts *>...>::type;
};

namespace detail_ {
// TODO
template <typename F, typename R, typename T0, typename T1>
F_CPU_GPU R Dispatch(F &&func, void *ptr, int index) {
  switch (index) {
    case 0:
      return func(reinterpret_cast<T0 *>(ptr));
    case 1:
      return func(reinterpret_cast<T1 *>(ptr));
    default:
      assert(false);
  }
}
}  // namespace detail_

/**
 * @brief A flat dynamic dispatcher as a replacement for inheritance
 * @see https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/taggedptr.h
 * @tparam Ts
 */
template <typename... Ts>
class Dispatcher {
public:
  using type_pack = TypePack<Ts...>;
  using ptr_type  = void *;

  F_CPU_GPU Dispatcher() = default;
  template <typename T>
  F_CPU_GPU Dispatcher(T *ptr)
      : m_index(IndexOf<std::remove_cv_t<T>, type_pack>::value), m_ptr(ptr) {}
  F_CPU_GPU Dispatcher(const Dispatcher &o)
      : m_index(o.m_index), m_ptr(o.m_ptr) {}
  F_CPU_GPU Dispatcher &operator=(const Dispatcher &o) {
    m_index = o.m_index, m_ptr = o.m_ptr;
  }

  F_CPU_GPU std::size_t index() const { return m_index; }
  template <typename T = void>
  F_CPU_GPU T *ptr() {
    return reinterpret_cast<T *>(m_ptr);
  }
  template <typename T = void>
  F_CPU_GPU T *ptr() const {
    return reinterpret_cast<const T *>(m_ptr);
  }

  /**
   * @brief Call and return func(ptr()), `func` is expected to be a wrapper
   * lambda of the original object, capturing all its parameters
   */
  template <typename F>
  F_CPU_GPU decltype(auto) dispatch(F &&func) {
    using R = typename ReturnType<F, Ts...>::type;
    return detail_::Dispatch<F, R, Ts...>(func, ptr(), index());
  }

protected:
  std::size_t m_index;
  ptr_type    m_ptr;
};

FLG_NAMESPACE_END

#endif
