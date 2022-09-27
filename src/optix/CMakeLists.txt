# This directory contains the GPU components of the renderer.
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
include(FindOptiX.cmake)

# CPMAddPackage("gh:eyalroz/cuda-api-wrappers@0.4.5")
add_library(fledge_optix managed_memory.cpp)
target_link_libraries(fledge_optix PUBLIC CUDA::cudart OptiX)
target_include_directories(fledge_optix PUBLIC ${CMAKE_SOURCE_DIR}/src/)
add_library(fledge::optix ALIAS fledge_optix)
