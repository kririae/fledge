# CPMAddPackage(
#   NAME embree
#   GITHUB_REPOSITORY embree/embree
#   GIT_TAG v3.13.4
#   VERSION 3.13.4
#   OPTIONS "EMBREE_TUTORIALS OFF"
# )
# set_property(TARGET embree PROPERTY CXX_STANDARD 17)

# https://github.com/embree/embree/issues/385
find_package(embree REQUIRED)

add_library(fledge_embree eprimitive.cpp)
target_include_directories(fledge_embree PUBLIC ${CMAKE_SOURCE_DIR}/src/)
target_link_libraries(fledge_embree PUBLIC embree miniply)
