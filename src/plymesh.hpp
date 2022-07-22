#ifndef __PLYMESH_HPP__
#define __PLYMESH_HPP__

#include <miniply.h>

#include <filesystem>
#include <memory>

#include "fwd.hpp"

SV_NAMESPACE_BEGIN

std::shared_ptr<TriangleMesh> make_TriangleMesh(const std::string &path);

SV_NAMESPACE_END
#endif
