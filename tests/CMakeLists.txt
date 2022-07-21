CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG release-1.12.1
  VERSION 1.12.1
  OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
)

add_executable(accel_tests accel_tests.cpp)
target_include_directories(accel_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(accel_tests
  gtest gtest_main
  smallvol_lib
)

add_test(test_all accel_tests)
