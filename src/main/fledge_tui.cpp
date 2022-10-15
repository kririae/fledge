#include "external/fxtui/tui.hpp"
#include "fledge.h"
#include "gpu/gpu_memory.hpp"
#include "gpu/gpu_rng.hpp"
#include "resource.hpp"
#include "rng.hpp"

using namespace fledge;
using namespace fledge::tui;

int main() {
  // WIP
  MakeWindow();

  Resource resource(GlobalManagedMemoryResource());
  auto     rng = MakeRandom(resource, 114514);
}
