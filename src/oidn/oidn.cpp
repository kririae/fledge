#include "oidn.hpp"

FLG_NAMESPACE_BEGIN

Film Denoise(const Film &pre_filtered) {
  using namespace ON;

  // Copy the buffers
  Film post_filtered = pre_filtered;

  // Acquire parameters from Film
  int width  = post_filtered.m_resX;
  int height = post_filtered.m_resY;

  OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
  oidnCommitDevice(device);

  void *colorPtr  = (void *)post_filtered.m_buffers[0].data();
  void *albedoPtr = (void *)post_filtered.m_buffers[1].data();
  void *normalPtr = (void *)post_filtered.m_buffers[2].data();
  void *outputPtr = (void *)post_filtered.m_buffers[3].data();

  // Create a filter for denoising a beauty (color) image using optional
  // auxiliary images too
  OIDNFilter filter =
      oidnNewFilter(device, "RT");  // generic ray tracing filter
  oidnSetSharedFilterImage(filter, "color", colorPtr, OIDN_FORMAT_FLOAT3, width,
                           height, 0, 0, 0);  // beauty
  oidnSetSharedFilterImage(filter, "albedo", albedoPtr, OIDN_FORMAT_FLOAT3,
                           width, height, 0, 0, 0);  // auxiliary
  oidnSetSharedFilterImage(filter, "normal", normalPtr, OIDN_FORMAT_FLOAT3,
                           width, height, 0, 0, 0);  // auxiliary
  oidnSetSharedFilterImage(filter, "output", outputPtr, OIDN_FORMAT_FLOAT3,
                           width, height, 0, 0, 0);  // denoised beauty
  oidnSetFilter1b(filter, "hdr", true);              // beauty image is HDR
  oidnCommitFilter(filter);
  // Filter the image
  oidnExecuteFilter(filter);

  // Check for errors
  const char *errorMessage;
  if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE)
    printf("Error: %s\n", errorMessage);

  // Cleanup
  oidnReleaseFilter(filter);
  oidnReleaseDevice(device);

  return post_filtered;
}

FLG_NAMESPACE_END