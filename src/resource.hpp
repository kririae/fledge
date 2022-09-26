#ifndef __RESOURCE_HPP__
#define __RESOURCE_HPP__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/scalable_allocator.h>

#include <boost/container/flat_map.hpp>
#include <functional>
#include <iterator>
#include <list>
#include <memory_resource>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "fledge.h"

FLG_NAMESPACE_BEGIN

namespace detail_ {
struct DestructorBase {
  // avoid linkage error
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
class managed_memory_resource : public std::pmr::memory_resource {
public:
  virtual ~managed_memory_resource() = default;
  void *allocate(std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t)) {
    return do_allocate(bytes, alignment);
  }

  void deallocate(void *p, std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) {
    return do_deallocate(p, bytes, alignment);
  }
  bool is_equal(const std::pmr::memory_resource &other) {
    return do_is_equal(other);
  }

  void *do_allocate(std::size_t bytes, std::size_t alignment) override {
    (void)alignment;
    void *ret;
    cudaMallocManaged(&ret, bytes);
    return ret;
  }
  void do_deallocate(void *p, std::size_t bytes,
                     std::size_t alignment) override {
    (void)bytes, (void)alignment;
    cudaFree(p);
  }
  bool do_is_equal(
      const std::pmr::memory_resource &other) const noexcept override {
    return this == &other;
  }
};
};  // namespace detail_

inline std::pmr::memory_resource *GlobalManagedMemoryResource() {
  static detail_::managed_memory_resource resource;
  return &resource;
}
/**
 * @brief The resource manager for the whole rendering system.
 */
struct Resource {
  Resource() = default;  // use local scalable_resource
  Resource(std::pmr::memory_resource *upstream)
      : m_upstream(upstream), m_mem_resource(&m_upstream) {}

  Resource(const Resource &)             = delete;
  Resource &operator()(const Resource &) = delete;
  virtual ~Resource() { release(); }

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
    assert(((size_t)(void *)mem) % Align == 0);
    allocator.construct(
        mem, std::forward<Args>(
                 args)...);  // instead of placement new and new_object
    m_destructors.emplace(static_cast<ptr_t>(mem),
                          new detail_::Destructor<T>(mem));
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
    assert(((size_t)(void *)mem) % Align == 0);
    if constexpr (sizeof...(args) == 0) {
      new (mem) T_[n];
    } else {
      for (size_t i = 0; i < n; ++i)
        allocator.construct(mem + i, std::forward<Args>(args)...);
    }  // if constexpr
    m_destructors.emplace(static_cast<ptr_t>(mem),
                          new detail_::ArrayDestructor<T>(mem, n));
    return mem;
  }

  void release() {
    for (auto &i : m_destructors) delete i.second;
    m_destructors.clear();
    m_mem_resource.release();
  }
  void printStat() {}

protected:
  using ptr_t = void *;
  boost::container::flat_map<ptr_t, detail_::DestructorBase *> m_destructors;
  tbb::cache_aligned_resource                                  m_upstream{
      oneapi::tbb::scalable_memory_resource()};
  std::pmr::unsynchronized_pool_resource m_mem_resource{&m_upstream};
};

FLG_NAMESPACE_END

#endif
