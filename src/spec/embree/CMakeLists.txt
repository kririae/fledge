# CPMAddPackage(
#   NAME embree
#   GITHUB_REPOSITORY embree/embree 
#   GIT_TAG v3.13.4
#   VERSION 3.13.4
#   OPTIONS "EMBREE_TUTORIALS OFF"
# )

# https://github.com/embree/embree/issues/385
# due to the removed std::allocator::destroy, embree cannot compile 
# on my machine
find_package(embree REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(fledge_embree eprimitive.cpp)
target_include_directories(fledge_embree PUBLIC ../../)
target_link_libraries(fledge_embree PUBLIC embree Eigen3::Eigen miniply)
