#include "optix_interface.hpp"

#include <optix_function_table_definition.h>

#include "fledge.h"
#include "optix7.h"

FLG_NAMESPACE_BEGIN
namespace optix {
void InitOptiX() {
  cudaFree(0);
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  if (num_devices == 0)
    throw std::runtime_error("no CUDA capable devices found!");
  SLog("found %d CUDA devices", num_devices);
  OPTIX_CHECK(optixInit());
}
}  // namespace optix
FLG_NAMESPACE_END
