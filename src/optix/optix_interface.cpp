#include "optix_interface.hpp"

#include <optix_function_table_definition.h>

#include "fledge.h"
#include "optix7.h"

FLG_NAMESPACE_BEGIN
namespace optix {
// Link against .*ptx_embedded.c
extern "C" char embedded_ptx_code[];
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
  alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
  alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
  alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  int objectID;
};

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
