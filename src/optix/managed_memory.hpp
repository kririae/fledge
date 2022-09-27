#ifndef __OPTIX_MEMORY_HPP__
#define __OPTIX_MEMORY_HPP__

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <memory_resource>

#include "fledge.h"

FLG_NAMESPACE_BEGIN

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

inline std::pmr::memory_resource *GlobalManagedMemoryResource() {
  static managed_memory_resource resource;
  return &resource;
}

FLG_NAMESPACE_END

#endif
