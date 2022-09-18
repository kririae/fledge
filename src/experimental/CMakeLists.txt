add_library(fledge_experimental base_bvh.cpp)
set_property(TARGET fledge_experimental PROPERTY CXX_STANDARD 20)
target_include_directories(fledge_experimental PUBLIC ${CMAKE_SOURCE_DIR}/src/)
target_link_libraries(fledge_experimental PRIVATE embree)

add_executable(bvh_test bvh_test.cpp)
target_link_libraries(bvh_test PRIVATE fledge_experimental fledge_lib)
set_target_properties(bvh_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/bin)

# Add nanobind
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND
  "${PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(bvh_test_py NB_STATIC bvh_test_py.cpp)
target_link_libraries(bvh_test_py PRIVATE fledge_experimental fledge_lib)