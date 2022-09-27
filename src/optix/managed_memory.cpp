#include "managed_memory.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "fledge.h"
#include "optix/optix7.h"

FLG_NAMESPACE_BEGIN
namespace optix {

void *managed_memory_resource::allocate(std::size_t bytes,
                                        std::size_t alignment) {
  return do_allocate(bytes, alignment);
}

void managed_memory_resource::deallocate(void *p, std::size_t bytes,
                                         std::size_t alignment) {
  return do_deallocate(p, bytes, alignment);
}

bool managed_memory_resource::is_equal(const std::pmr::memory_resource &other) {
  return do_is_equal(other);
}

void *managed_memory_resource::do_allocate(std::size_t bytes,
                                           std::size_t alignment) {
  (void)alignment;
  void *ret;
  cudaMallocManaged(&ret, bytes);
  return ret;
}

void managed_memory_resource::do_deallocate(void *p, std::size_t bytes,
                                            std::size_t alignment) {
  (void)bytes, (void)alignment;
  cudaFree(p);
}

bool managed_memory_resource::do_is_equal(
    const std::pmr::memory_resource &other) const noexcept {
  return this == &other;
}
std::pmr::memory_resource *GlobalManagedMemoryResource() {
  static managed_memory_resource resource;
  return &resource;
}

}  // namespace optix
FLG_NAMESPACE_END