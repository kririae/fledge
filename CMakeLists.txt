cmake_minimum_required(VERSION 3.23)
project(smallvol)

# project options
option(USE_TBB "use TBB instead of OpenMP for parallization" ON)

set(CMAKE_CXX_STANDARD 20)

include("get_cpm.cmake")
add_subdirectory(src/)
