set(CMAKE_CXX_STANDARD 20)

CPMAddPackage("gh:vilya/miniply#master")

if(miniply_ADDED)
  set(miniplySources ${miniply_SOURCE_DIR}/miniply.cpp)
  add_library(miniply ${miniplySources})
  set_property(TARGET miniply PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_include_directories(miniply PUBLIC ${miniply_SOURCE_DIR})
endif()

find_package(Eigen3 REQUIRED)
find_package(OpenImageIO REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP)
find_package(OpenVDB REQUIRED COMPONENTS openvdb)

# add it before the compiler specifications
add_subdirectory(spec/oidn/)

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
        -Wno-unused-function -Wno-unused-private-field -Wno-error=deprecated-declarations -Wno-gnu-zero-variadic-macro-arguments -march=native -mtune=native")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -g -Wall -Wextra \
        -Wpedantic -Werror -Wno-unused -Wno-error=deprecated-declarations -Wno-error=maybe-uninitialized -march=native -mtune=native")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif()

add_subdirectory(spec/embree/)

file(GLOB source "${PROJECT_SOURCE_DIR}/src/*.c*")
file(GLOB all_source "${PROJECT_SOURCE_DIR}/src/*.[ch]pp")
file(GLOB_RECURSE cmake_files "${PROJECT_SOURCE_DIR}/CMakeLists.txt")

# clang-format
clang_format(format_fledge "${all_source}")
cmake_format(format_cmake "${cmake_files}")

add_library(fledge_lib "${source}")
add_executable(fledge "${source}")

if(USE_ISPC)
  add_subdirectory(spec/ispc/)
  target_link_libraries(fledge_lib PUBLIC fledge_lib_ispc)
  target_compile_definitions(fledge_lib PUBLIC -DUSE_ISPC)
endif()

target_link_libraries(fledge_lib PUBLIC Eigen3::Eigen OpenVDB::openvdb
  OpenImageIO::OpenImageIO TBB::tbb miniply fledge_embree fledge_lib_oidn)
target_link_libraries(fledge PUBLIC fledge_lib)
set_target_properties(fledge PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/bin)

if(USE_TBB)
  target_compile_definitions(fledge PUBLIC -DUSE_TBB)
endif()

# if (USE_EMBREE)
target_compile_definitions(fledge PUBLIC -DUSE_EMBREE)

# endif()
