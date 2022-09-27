set(CMAKE_CXX_STANDARD 20)

CPMAddPackage("gh:vilya/miniply#master")

if(miniply_ADDED)
  set(miniplySources ${miniply_SOURCE_DIR}/miniply.cpp)
  add_library(miniply ${miniplySources})
  set_property(TARGET miniply PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_include_directories(miniply PUBLIC ${miniply_SOURCE_DIR})
endif()

# CPMAddPackage("gh:eyalroz/cuda-api-wrappers@0.4.5")
find_package(Thrust REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
find_package(OpenImageIO REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP)
find_package(OpenVDB REQUIRED COMPONENTS openvdb)

# add fmt
CPMAddPackage("gh:fmtlib/fmt#9.1.0")

# add it before the compiler specifications
add_subdirectory(external/oidn/)

# add OpenMP
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# compilers' specification
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -pipe -g -Wall -Wextra \
        -Wpedantic -Werror -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable \
        -Wno-unused-function -Wno-unused-private-field -Wno-error=deprecated-declarations -Wno-gnu-zero-variadic-macro-arguments -Wno-extra-semi -Wno-zero-length-array -march=native -mtune=native")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -g -Wall -Wextra \
        -Wpedantic -Werror -Wno-unused -Wno-error=deprecated-declarations -Wno-error=maybe-uninitialized -march=native -mtune=native")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif()

add_subdirectory(external/embree/)

file(GLOB source "${PROJECT_SOURCE_DIR}/src/*.c*")
file(GLOB all_source "${PROJECT_SOURCE_DIR}/src/*.[ch]pp")
file(GLOB_RECURSE cmake_files "${PROJECT_SOURCE_DIR}/CMakeLists.txt")

add_subdirectory(experimental/)

# clang-format
clang_format(format_fledge "${all_source}")
cmake_format(format_cmake "${cmake_files}")

add_library(fledge_lib "${source}")
set_property(TARGET fledge_lib PROPERTY CXX_STANDARD 20)
add_executable(fledge "${source}")

if(FLEDGE_USE_ISPC)
  add_subdirectory(external/ispc/)
  target_link_libraries(fledge_lib PUBLIC fledge_lib_ispc)
  target_compile_definitions(fledge_lib PUBLIC -DFLEDGE_USE_ISPC)
endif()

if(FLEDGE_USE_JEMALLOC)
  # Introduct jemalloc
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(JEMALLOC jemalloc)
  pkg_search_module(JEMALLOC REQUIRED jemalloc)
  target_include_directories(fledge_lib PUBLIC ${JEMALLOC_INCLUDE_DIRS})
  target_link_libraries(fledge_lib PUBLIC ${JEMALLOC_LIBRARIES})
endif()

add_subdirectory(materials/)

target_link_libraries(fledge_lib PUBLIC OpenVDB::openvdb CUDA::cudart Thrust::Thrust fmt
  OpenImageIO::OpenImageIO TBB::tbb TBB::tbbmalloc miniply fledge_embree fledge_lib_oidn fledge_materials)
target_link_libraries(fledge PUBLIC fledge_lib)
set_target_properties(fledge PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/bin)

# if(FLEDGE_USE_TBB)
# target_compile_definitions(fledge PUBLIC -DFLEDGE_USE_TBB)
# endif()
target_compile_definitions(fledge PUBLIC -DFLEDGE_USE_EMBREE)
