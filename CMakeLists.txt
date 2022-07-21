cmake_minimum_required(VERSION 3.23)
project(smallvol)

# project options
option(USE_TBB "use TBB instead of OpenMP for parallization" OFF)
set(USE_SANITIZER
    ""
    CACHE
      STRING
      "Compile with a sanitizer. Options are: Address, Memory, MemoryWithOrigins, Undefined, Thread, Leak, 'Address;Undefined', CFI"
)

set(CMAKE_CXX_STANDARD 20)

include("get_cpm.cmake")
enable_testing()

add_subdirectory(src/)
add_subdirectory(tests/)
