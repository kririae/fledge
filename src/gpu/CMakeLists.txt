enable_language(CUDA)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(fledge_gpu gpu_memory.cpp gpu_rng.cu)
target_link_libraries(fledge_gpu PUBLIC CUDA::cudart)
target_include_directories(fledge_gpu PUBLIC "${CMAKE_SOURCE_DIR}/src/")
add_library(fledge::gpu ALIAS fledge_gpu)