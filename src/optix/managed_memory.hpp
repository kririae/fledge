#ifndef __OPTIX_MEMORY_HPP__
#define __OPTIX_MEMORY_HPP__

#include <memory_resource>

#include "fledge.h"

FLG_NAMESPACE_BEGIN
namespace optix {

std::pmr::memory_resource *GlobalManagedMemoryResource();

class managed_memory_resource : public std::pmr::memory_resource {
public:
  virtual ~managed_memory_resource() = default;
  void *allocate(std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t));
  void  deallocate(void *p, std::size_t bytes,
                   std::size_t alignment = alignof(std::max_align_t));
  bool  is_equal(const std::pmr::memory_resource &other);
  void *do_allocate(std::size_t bytes, std::size_t alignment) override;
  void  do_deallocate(void *p, std::size_t bytes,
                      std::size_t alignment) override;
  bool  do_is_equal(
       const std::pmr::memory_resource &other) const noexcept override;
};

}  // namespace optix
FLG_NAMESPACE_END

#endif
