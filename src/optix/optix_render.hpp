#include "debug.hpp"
#include "fledge.h"
#include "optix/optix7.h"
#include "optix/optix_interface.hpp"
#include "render.hpp"

FLG_NAMESPACE_BEGIN
namespace optix {
/**
 * @brief OptiXRender serves as the bridge between the general scene description
 * and the new integrators
 */
class OptiXRender : public RenderBase {
public:
  OptiXRender(Scene *scene);
  void init() override;
  bool preProcess() override;
  bool saveImage(const std::string &name, bool denoise = false) override;
  bool render() override;
  EBackendType getBackends() override { return EBackendType::EOptiXBackend; }

private:
  static void contextLogCb(unsigned int level, const char *tag,
                           const char *message, void *) {
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
  }

  // OptiX helper functions
  void initOptiX();
  void createContext();

  // Internal Objects
  Scene *m_scene;

  // CUDA Objects
  CUcontext      cudaContext;
  CUstream       stream;
  cudaDeviceProp deviceProps;

  // OptiX Objects
  OptixDeviceContext optixContext;
};
}  // namespace optix
FLG_NAMESPACE_END
