cpmaddpackage("gh:google/googletest#release-1.12.1")

find_package(googletest)

add_executable(accel_tests accel_tests.cpp)
target_include_directories(accel_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(accel_tests
  GTest::gtest_main
  smallvol_lib
)

add_test(test_all accel_tests)
