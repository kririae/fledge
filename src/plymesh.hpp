#ifndef __PLYMESH_HPP__
#define __PLYMESH_HPP__

#include <miniply.h>

#include <filesystem>
#include <memory>

#include "fledge.h"
#include "debug.hpp"
#include "common/vector.h"

FLG_NAMESPACE_BEGIN

std::shared_ptr<TriangleMesh> MakeTriangleMesh(const std::string &path);
std::shared_ptr<TriangleMesh> MakeMeshedSphere();

FLG_NAMESPACE_END
#endif
