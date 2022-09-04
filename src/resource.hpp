#ifndef __RESOURCE_HPP__
#define __RESOURCE_HPP__

#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/scalable_allocator.h>

// clang-format off
// #include <span>
// #include <cuda/api.hpp>
// clang-format on

#include <functional>
#include <iterator>
#include <list>
#include <memory_resource>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "fledge.h"

FLG_NAMESPACE_BEGIN

namespace detail_ {
struct DestructorBase {
  virtual ~DestructorBase() {}
};
template <typename T>
requires(!std::is_array_v<T>) struct Destructor : public DestructorBase {
  Destructor(T *p) : m_p(p) {}
  ~Destructor() override { std::destroy_at(m_p); }
  T *m_p;
};
template <typename T, typename T_ = std::remove_extent_t<T>>
requires(std::is_unbounded_array_v<T>) struct ArrayDestructor
    : public DestructorBase {
  ArrayDestructor(T_ *p, size_t n) : m_p(p), m_n(n) {}
  ~ArrayDestructor() override { std::destroy_n(m_p, m_n); }
  T_    *m_p;
  size_t m_n;
};
};  // namespace detail_

/**
 * @brief The resource manager for the whole rendering system.
 */
struct Resource {
  Resource() = default;  // use local scalable_resource
  Resource(std::pmr::memory_resource *upstream)
      : m_upstream(upstream), m_mem_resource(&m_upstream) {}

  Resource(const Resource &)             = delete;
  Resource &operator()(const Resource &) = delete;
  ~Resource() { release(); }

  /**
   * @brief Allocate memory using the memory resource manager and the
   * constructor parameters
   * @note The interface is exactly the same as make_unique since C++ 14.
   * @tparam T
   * @tparam Args
   * @return T* A pointer to the allocated memory fragment
   */
  template <typename T, size_t Align = alignof(T), typename... Args>
  std::enable_if_t<!std::is_array<T>::value, T *> alloc(Args &&...args) {
    auto allocator = std::pmr::polymorphic_allocator<T>(&m_mem_resource);
    T   *mem = static_cast<T *>(allocator.allocate_bytes(sizeof(T), Align));
    assert(static_cast<size_t>(mem) % Align == 0);
    allocator.construct(
        mem, std::forward<Args>(
                 args)...);  // instead of placement new and new_object
    m_destructors.push_back(new detail_::Destructor<T>(mem));
    return mem;
  }

  /**
   * @brief Allocate array using the memory resource manager and the size of the
   * array
   * @see The previous `alloc`
   * @note The implementation is quite different from unique_ptr, which accepts
   * a constructor for every element in the array.
   *
   * @tparam T
   * @tparam Args
   * @param n The number of elements in the array.
   * @param args
   * @return T* A pointer to the allocated memory fragment
   */
  template <typename T, size_t Align = alignof(T),
            typename T_ = std::remove_extent_t<T>, typename... Args>
  std::enable_if_t<std::is_unbounded_array_v<T>, T_ *> alloc(size_t n,
                                                             Args &&...args) {
    auto allocator = std::pmr::polymorphic_allocator<T_>(&m_mem_resource);
    T_  *mem =
        static_cast<T_ *>(allocator.allocate_bytes(sizeof(T_) * n, Align));
    assert(static_cast<size_t>(mem) % Align == 0);
    if constexpr (sizeof...(args) == 0) {
      new (mem) T_[n];
    } else {
      for (size_t i = 0; i < n; ++i)
        allocator.construct(mem + i, std::forward<Args>(args)...);
    }  // if constexpr
    m_destructors.push_back(new detail_::ArrayDestructor<T>(mem, n));
    return mem;
  }

  void release() {
    for (auto &i : m_destructors) delete i;
    m_destructors.clear();
    m_mem_resource.release();
  }
  void printStat() {}

private:
  std::list<detail_::DestructorBase *> m_destructors;
  tbb::cache_aligned_resource          m_upstream{
      oneapi::tbb::scalable_memory_resource()};
  std::pmr::unsynchronized_pool_resource m_mem_resource{&m_upstream};
};

FLG_NAMESPACE_END

#endif
