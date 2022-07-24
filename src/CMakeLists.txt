set(CMAKE_CXX_STANDARD 20)

CPMAddPackage("gh:StableCoder/cmake-scripts#22.01")
include(${cmake-scripts_SOURCE_DIR}/sanitizers.cmake)
include(${cmake-scripts_SOURCE_DIR}/formatting.cmake)

CPMAddPackage("gh:vilya/miniply#master")

if (miniply_ADDED)
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
        -Wpedantic -Werror -Wno-unused-parameter -Wno-unused-variable \
        -Wno-unused-function -Wno-gnu-zero-variadic-macro-arguments \
        -march=native -mtune=native")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -g -Wall -Wextra \
        -Wpedantic -Werror -Wno-unused -march=native -mtune=native")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")

endif()

add_subdirectory(embree/)

file(GLOB source "${PROJECT_SOURCE_DIR}/src/*.c*")
file(GLOB all_source "${PROJECT_SOURCE_DIR}/src/*.[ch]pp")
file(GLOB_RECURSE cmake_files "${PROJECT_SOURCE_DIR}/CMakeLists.txt")

# clang-format
clang_format(format_smallvol "${all_source}")
cmake_format(format_cmake "${cmake_files}")

add_library(smallvol_lib "${source}")
add_executable(smallvol "${source}")

target_link_libraries(smallvol_lib PUBLIC Eigen3::Eigen OpenVDB::openvdb
                                      OpenImageIO::OpenImageIO TBB::tbb miniply smallvol_embree)
target_link_libraries(smallvol PUBLIC smallvol_lib)
set_target_properties(smallvol PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                          ${CMAKE_BINARY_DIR}/bin)
if(USE_TBB)
  target_compile_definitions(smallvol PUBLIC -DUSE_TBB)
endif()
# if (USE_EMBREE)
  target_compile_definitions(smallvol PUBLIC -DUSE_EMBREE)
# endif()
