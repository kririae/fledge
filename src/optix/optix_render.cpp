#include "optix_render.hpp"

#include "optix7.h"
#include "optix_interface.hpp"

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

OptiXRender::OptiXRender(Scene *scene) : m_scene(scene) {
  OLog("OptiX Renderer initialized");
  initOptiX();

  OLog("creating OptiX context...");
  createContext();
}

void OptiXRender::init() {}

bool OptiXRender::preProcess() {
  return true;
}

bool OptiXRender::saveImage(const std::string &name, bool denoise) {
  return true;
}

bool OptiXRender::render() {
  return true;
}

void OptiXRender::initOptiX() {
  InitOptiX();  // from optix_interface.hpp
}

void OptiXRender::createContext() {
  const int deviceID = 0;
  CUDA_CHECK(cudaSetDevice(deviceID));
  CUDA_CHECK(cudaStreamCreate(&stream));

  cudaGetDeviceProperties(&deviceProps, deviceID);
  OLog("executing on device {}", deviceProps.name);

  CUresult cuRes = cuCtxGetCurrent(&cudaContext);
  if (cuRes != CUDA_SUCCESS)
    OErr("Error querying current context: error code {}\n", cuRes);

  OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
  OPTIX_CHECK(
      optixDeviceContextSetLogCallback(optixContext, contextLogCb, nullptr, 4));
}

}  // namespace optix
FLG_NAMESPACE_END
