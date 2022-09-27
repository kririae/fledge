CPMAddPackage(
  NAME embree
  GITHUB_REPOSITORY embree/embree
  GIT_TAG v3.13.5-alpha.0
  VERSION 3.13.5
  OPTIONS
  "EMBREE_TUTORIALS OFF"
  "EMBREE_GEOMETRY_QUAD OFF"
  "EMBREE_GEOMETRY_CURVE OFF"
  "EMBREE_GEOMETRY_SUBDIVISION OFF"
  "EMBREE_GEOMETRY_INSTANCE OFF"
  "EMBREE_GEOMETRY_POINT OFF"
)
set_property(TARGET embree PROPERTY CXX_STANDARD 17)

# https://github.com/embree/embree/issues/385
# find_package(embree REQUIRED)
add_library(fledge_lib_embree eprimitive.cpp)
target_include_directories(fledge_lib_embree PUBLIC ${CMAKE_SOURCE_DIR}/src/)
target_link_libraries(fledge_lib_embree PUBLIC embree miniply)

add_library(fledge::lib::embree ALIAS fledge_lib_embree)
