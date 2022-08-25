#ifndef __OIDN_HPP__
#define __OIDN_HPP__

#include <OpenImageDenoise/oidn.h>

#include "../film.hpp"
#include "../fwd.hpp"

FLG_NAMESPACE_BEGIN

/* *
 * return a new denoised film
 * Naive implementation from https://github.com/OpenImageDenoise/oidn
 */
Film Denoise(const Film &pre_filtered);

FLG_NAMESPACE_END

#endif