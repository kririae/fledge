#ifndef __PLYMESH_HPP__
#define __PLYMESH_HPP__

#include <miniply.h>

#include <filesystem>
#include <memory>

#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"
#include "resource.hpp"

FLG_NAMESPACE_BEGIN

TriangleMesh *MakeTriangleMesh(const std::string &path, Resource &resource);
TriangleMesh *MakeMeshedSphere(int n_theta, int n_phi, Float radius,
                               Resource &resource);

FLG_NAMESPACE_END
#endif
