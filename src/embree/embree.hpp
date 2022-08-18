#include <embree3/rtcore.h>
#include <embree3/rtcore_device.h>

#include <iostream>

#include "debug.hpp"
#include "fwd.hpp"
#include "vector.hpp"

SV_NAMESPACE_BEGIN

static void embreeErrorFunction([[maybe_unused]] void *userPtr,
                                enum RTCError error, const char *str) {
  SLog("embree error %d: %s\n", error, str);
}

inline RTCDevice embreeInitializeDevice() {
  RTCDevice device = rtcNewDevice(nullptr);
  if (!device)
    SLog("embree error %d: cannot create device\n", rtcGetDeviceError(nullptr));
  rtcSetDeviceErrorFunction(device, embreeErrorFunction, nullptr);
  return device;
}

SV_NAMESPACE_END