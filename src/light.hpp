#ifndef __LIGHT_HPP__
#define __LIGHT_HPP__

#include "fwd.hpp"

SV_NAMESPACE_BEGIN

class Light {
public:
  virtual ~Light();
  virtual Vector3f sampleLi() = 0;
};

SV_NAMESPACE_END

#endif
