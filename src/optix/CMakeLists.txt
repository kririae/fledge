# This directory contains the GPU components of the renderer.
include(FindOptiX.cmake)
include(configure_optix.cmake)

include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
include_directories("${OptiX_INCLUDE}") # set by FindOptiX.cmake
include_directories("${CMAKE_SOURCE_DIR}/src/")

# CPMAddPackage("gh:eyalroz/cuda-api-wrappers@0.4.5")
cuda_compile_and_embed(embedded_ptx_code optix_kernel.cu)
add_library(fledge_optix managed_memory.cpp optix_interface.cpp optix_render.cpp ${embedded_ptx_code})
target_link_libraries(fledge_optix PUBLIC CUDA::cudart CUDA::cuda_driver OptiX)
add_library(fledge::optix ALIAS fledge_optix)
