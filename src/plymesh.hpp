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
std::shared_ptr<TriangleMesh> MakeMeshedSphere();

FLG_NAMESPACE_END
#endif
