#ifndef __RESOURCE_HPP__
#define __RESOURCE_HPP__

#include <cstdint>
#include <functional>
#include <list>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "fledge.h"

FLG_NAMESPACE_BEGIN

namespace detail {
struct DefaultAllocator {
  void *operator()(size_t size) { return std::malloc(size); }
};
struct DefaultDeallocator {
  void operator()(void *ptr) { std::free(ptr); }
};
template <size_t N = 16>
struct AlignedAllocator {
  void *operator()(size_t size) { return std::aligned_alloc(N, size); }
};
};  // namespace detail

/**
 * @brief The resource manager for the whole rendering system.
 */
template <typename Allocator, typename Deallocator>
struct GenericResource {
  GenericResource()                                    = default;
  GenericResource(const GenericResource &)             = delete;
  GenericResource &operator()(const GenericResource &) = delete;
  ~GenericResource() {
    for (auto &i : m_blocks) {
      for (size_t idx = 0; idx < i.m_n; ++idx)
        i.m_destruct((uint8_t *)i.m_ptr +
                     idx * i.m_size);  // traverse to call the destructor
      Deallocator()(i.m_ptr);
    }
  }

  /**
   * @brief Allocate memory using the memory resource manager and the
   * constructor parameters
   * @note The interface is exactly the same as make_unique since C++ 14.
   * @tparam T
   * @tparam Args
   * @return T* A pointer to the allocated memory fragment
   */
  template <typename T, typename... Args>
  std::enable_if_t<!std::is_array<T>::value, T *> alloc(Args &&...args) {
    MemoryBlock block;  // create a new block for this allocation
    block.m_description = typeid(T).name();
    block.m_ptr         = Allocator()(sizeof(T));
    assert(block.m_ptr != nullptr);
    block.m_size = sizeof(T);
    block.m_n    = 1;
    new (block.m_ptr)
        T(std::forward<Args>(args)...);  // assign the object to the block
    block.m_destruct = [](const void *x) { static_cast<const T *>(x)->~T(); };
    m_blocks.push_back(block);  // register the block
    return static_cast<T *>(block.m_ptr);
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
  template <typename T, typename T_ = std::remove_extent_t<T>, typename... Args>
  std::enable_if_t<std::is_unbounded_array_v<T>, T_ *> alloc(size_t n,
                                                             Args &&...args) {
    MemoryBlock block;  // yet, create a new block
    block.m_description = std::string(typeid(T_).name()) + "__array";
    block.m_ptr =
        Allocator()(sizeof(T_) * n);  // use the default malloc for now
    assert(block.m_ptr != nullptr);
    block.m_size = sizeof(T_);
    block.m_n    = n;
    if constexpr (sizeof...(args) == 0) {
      new (block.m_ptr) T_[n]();  // call the constructors
    } else {
      for (size_t i = 0; i < n; ++i)  // initialize the elements one by one
        new ((uint8_t *)block.m_ptr + sizeof(T_) * i)
            T_(std::forward<Args>(args)...);
    }  // if constexpr
    block.m_destruct = [](const void *x) { static_cast<const T_ *>(x)->~T_(); };
    m_blocks.push_back(block);
    return static_cast<T_ *>(block.m_ptr);
  }

private:
  /**
   * @brief The smallest granularity of memory management in the system
   */
  struct MemoryBlock {
    void       *m_ptr;
    size_t      m_size, m_n;
    std::string m_description;
    // https://herbsutter.com/2016/09/25/to-store-a-destructor/
    void (*m_destruct)(const void *);
  };

  std::list<MemoryBlock> m_blocks;
};

using Resource =
    GenericResource<detail::DefaultAllocator, detail::DefaultDeallocator>;

FLG_NAMESPACE_END

#endif
