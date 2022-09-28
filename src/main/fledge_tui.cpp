#include "external/fxtui/tui.hpp"
#include "fledge.h"
#include "gpu/gpu_memory.hpp"
#include "gpu/gpu_rng.hpp"
#include "resource.hpp"
#include "rng.hpp"

using namespace fledge;
using namespace fledge::tui;

template <typename... Args>
inline Random MakeRandom(Resource &resource, Args &&...args) {
  RandomCPU *ptr_cpu = resource.alloc<RandomCPU>(std::forward<Args>(args)...);
  RandomGPU *ptr_gpu = resource.alloc<RandomGPU>(std::forward<Args>(args)...);
  return {ptr_cpu, ptr_gpu};
}

int main() {
  // WIP
  MakeWindow();

  Resource resource(GlobalManagedMemoryResource());
  auto     rng = MakeRandom(resource, 114514);
}
