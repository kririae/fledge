cmake_minimum_required(VERSION 3.23)
project(fledge LANGUAGES CXX C ISPC)
cmake_policy(SET CMP0135 NEW)

# project options
option(USE_TBB "use TBB instead of OpenMP for parallization" OFF)
option(USE_ISPC "use ISPC for SIMD execution" ON)
option(USE_EMBREE "use Embree 3 as the acceleration structure" ON)
option(USE_OIDN "use OIDN to denoise" ON)
set(USE_SANITIZER
  ""
  CACHE
  STRING
  "Compile with a sanitizer. Options are: Address, Memory, MemoryWithOrigins, Undefined, Thread, Leak, 'Address;Undefined', CFI"
)

set(CMAKE_CXX_STANDARD 20)

include("get_cpm.cmake")
add_subdirectory(src/)

enable_testing()
add_subdirectory(tests/)
