set(CMAKE_CXX_STANDARD 20)

function(flg_add_ext name)
  add_subdirectory(external/${name})
  target_link_libraries(fledge_lib PUBLIC fledge::lib::${name})
endfunction()

CPMAddPackage("gh:vilya/miniply#master")

if(miniply_ADDED)
  set(miniplySources ${miniply_SOURCE_DIR}/miniply.cpp)
  add_library(miniply ${miniplySources})
  set_property(TARGET miniply PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_include_directories(miniply PUBLIC ${miniply_SOURCE_DIR})
endif()

find_package(Thrust REQUIRED CONFIG)
find_package(OpenImageIO REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP)
find_package(OpenVDB REQUIRED COMPONENTS openvdb)

# add fmt
CPMAddPackage("gh:fmtlib/fmt#9.1.0")

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
        -Wno-unused-function -Wno-unused-private-field -Wno-error=deprecated-declarations -Wno-gnu-zero-variadic-macro-arguments -Wno-extra-semi -Wno-zero-length-array -Wno-nested-anon-types -Wno-gnu-anonymous-struct -march=native -mtune=native")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -g -Wall -Wextra \
        -Wpedantic -Werror -Wno-unused -Wno-error=deprecated-declarations -Wno-error=maybe-uninitialized -march=native -mtune=native")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif()

file(GLOB source "${PROJECT_SOURCE_DIR}/src/*.c*")
file(GLOB all_source "${PROJECT_SOURCE_DIR}/src/*.[ch]pp")
file(GLOB_RECURSE cmake_files "${PROJECT_SOURCE_DIR}/CMakeLists.txt")

add_subdirectory(experimental/)
add_subdirectory(gpu/)

# clang-format
clang_format(format_fledge "${all_source}")
cmake_format(format_cmake "${cmake_files}")

add_library(fledge_lib "${source}")
add_library(fledge::lib ALIAS fledge_lib)
set_property(TARGET fledge_lib PROPERTY CXX_STANDARD 20)

add_executable(fledge main/fledge.cpp)
add_executable(fledge_tui main/fledge_tui.cpp)

if(FLEDGE_USE_JEMALLOC)
  # Introduct jemalloc
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(JEMALLOC jemalloc)
  pkg_search_module(JEMALLOC REQUIRED jemalloc)
  target_include_directories(fledge_lib PUBLIC ${JEMALLOC_INCLUDE_DIRS})
  target_link_libraries(fledge_lib PUBLIC ${JEMALLOC_LIBRARIES})
endif()

add_subdirectory(materials/)

if(FLEDGE_USE_ISPC)
  target_compile_definitions(fledge_lib PUBLIC -DFLEDGE_USE_ISPC)
  flg_add_ext(ispc)
endif()

flg_add_ext(oidn)
flg_add_ext(fxtui)
flg_add_ext(embree)

add_subdirectory(optix/)

target_link_libraries(fledge_lib PUBLIC OpenVDB::openvdb Thrust::Thrust fmt
  OpenImageIO::OpenImageIO TBB::tbb TBB::tbbmalloc miniply fledge::materials fledge::optix fledge::gpu)

target_link_libraries(fledge PUBLIC fledge::lib)
set_target_properties(fledge PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(fledge_tui PUBLIC fledge::lib)
set_target_properties(fledge_tui PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/bin)

# if(FLEDGE_USE_TBB)
# target_compile_definitions(fledge PUBLIC -DFLEDGE_USE_TBB)
# endif()
target_compile_definitions(fledge PUBLIC -DFLEDGE_USE_EMBREE)
