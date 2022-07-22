CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG release-1.12.1
  VERSION 1.12.1
  OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
)

macro(sv_add_test name)
  add_executable(${name}_exe ${name}.cpp)
  target_include_directories(${name}_exe PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(${name}_exe
    gtest gtest_main
    smallvol_lib
  )
  add_test(${name} ${name}_exe)
endmacro()

sv_add_test(accel_tests)
sv_add_test(shape_tests)
sv_add_test(embree_tests)
