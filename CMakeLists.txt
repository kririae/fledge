cmake_minimum_required(VERSION 3.23)
project(fledge LANGUAGES CXX C)
cmake_policy(SET CMP0135 NEW)

# project options
option(FLEDGE_USE_TBB "use TBB instead of OpenMP for parallization" ON)
option(FLEDGE_USE_ISPC "use ISPC for SIMD execution" ON)
option(FLEDGE_USE_EMBREE "use Embree 3 as the acceleration structure" ON)
option(FLEDGE_USE_OIDN "use OIDN to denoise" ON)
option(FLEDGE_USE_JEMALLOC "use JEMALLOC for memory optimization" ON)
set(USE_SANITIZER
  ""
  CACHE
  STRING
  "Compile with a sanitizer. Options are: Address, Memory, MemoryWithOrigins, Undefined, Thread, Leak, 'Address;Undefined', CFI"
)

set(CMAKE_CXX_STANDARD 20)
include("get_cpm.cmake")

# Add sanitizer
CPMAddPackage("gh:StableCoder/cmake-scripts#22.01")
include(${cmake-scripts_SOURCE_DIR}/sanitizers.cmake)
include(${cmake-scripts_SOURCE_DIR}/formatting.cmake)

add_subdirectory(src/)

enable_testing()
add_subdirectory(tests/)
