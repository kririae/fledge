#ifndef __OPTIX7_H__
#define __OPTIX7_H__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <memory_resource>

#include "debug.hpp"
#include "fledge.h"
#include "fmt/core.h"

FLG_NAMESPACE_BEGIN
namespace optix {
#define CUDA_CALL(expr)                                        \
  {                                                            \
    cudaError_t err = expr;                                    \
    if (err != cudaSuccess) {                                  \
      fmt::print("CUDA Error: {}\n", cudaGetErrorString(err)); \
    }                                                          \
  }

#define OPTIX_CHECK(call)                                                  \
  {                                                                        \
    OptixResult res = call;                                                \
    if (res != OPTIX_SUCCESS) {                                            \
      fmt::print("Optix call ({}) failed with code {} (line {})\n", #call, \
                 res, __LINE__);                                           \
      exit(2);                                                             \
    }                                                                      \
  }

inline void InitOptiX() {
  cudaFree(0);
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  if (num_devices == 0)
    throw std::runtime_error("no CUDA capable devices found!");
  SLog("found %d CUDA devices\n", num_devices);
  OPTIX_CHECK(optixInit());
}
}  // namespace optix
FLG_NAMESPACE_END

#endif
