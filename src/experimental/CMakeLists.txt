add_library(fledge_experimental bvh.cpp)
set_property(TARGET fledge_experimental PROPERTY CXX_STANDARD 20)
target_include_directories(fledge_experimental PUBLIC ${CMAKE_SOURCE_DIR}/src/)
target_link_libraries(fledge_experimental PRIVATE embree)

add_executable(bvh_test bvh_test.cpp)
target_link_libraries(bvh_test PRIVATE fledge_experimental fledge_lib)
set_target_properties(bvh_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/bin)