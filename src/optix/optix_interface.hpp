#ifndef __OPTIX_INTERFACE_HPP__
#define __OPTIX_INTERFACE_HPP__

#include "fledge.h"

FLG_NAMESPACE_BEGIN
namespace optix {
// Link against .*ptx_embedded.c
extern "C" char embedded_ptx_code[];

void InitOptiX();

}  // namespace optix
FLG_NAMESPACE_END
#endif
