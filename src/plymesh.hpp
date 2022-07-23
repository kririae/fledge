#ifndef __PLYMESH_HPP__
#define __PLYMESH_HPP__

#include <miniply.h>

#include <filesystem>
#include <memory>

#include "fwd.hpp"

SV_NAMESPACE_BEGIN

std::shared_ptr<TriangleMesh> MakeTriangleMesh(const std::string &path);
std::shared_ptr<TriangleMesh> MakeMeshedSphere();

SV_NAMESPACE_END
#endif
