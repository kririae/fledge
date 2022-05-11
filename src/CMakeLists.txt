find_package(Eigen3 REQUIRED)
find_package(OpenVDB REQUIRED)
find_package(OpenImageIO REQUIRED)
find_package(TBB REQUIRED)

# add OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB source
        "${PROJECT_SOURCE_DIR}/src/*.c*")
add_executable(smallvol "${source}")

target_link_libraries(smallvol PUBLIC 
        Eigen3::Eigen OpenVDB::openvdb OpenImageIO::OpenImageIO TBB::tbb)
# target_link_libraries(smallvol PUBLIC 
#         Eigen3::Eigen OpenImageIO::OpenImageIO TBB::tbb)
set_target_properties(smallvol
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if (USE_TBB)
        target_compile_definitions(smallvol PUBLIC -DUSE_TBB)
endif()

# compilers' specification
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -g -Wpedantic \
        -Werror -Wno-unused -Wall -Wextra -march=native -mtune=native")
        add_compile_options(-fsanitize=address -fsanitize=undefined)
        add_link_options(-fsanitize=address -fsanitize=undefined)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -g -Wpedantic \
        -Werror -Wno-unused -Wall -Wextra -march=native -mtune=native")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif()
