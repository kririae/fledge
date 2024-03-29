CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG release-1.12.1
  VERSION 1.12.1
  OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
)

enable_language(CUDA)

macro(flg_add_test name)
  add_executable(${name}_exe ${name}.cpp)
  target_include_directories(${name}_exe PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(${name}_exe
    gtest gtest_main
    fledge::lib
  )
  add_test(${name} ${name}_exe)
endmacro()

macro(flg_add_cuda_test name)
  add_executable(${name}_cuda_exe ${name}.cu)
  target_include_directories(${name}_cuda_exe PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(${name}_cuda_exe
    gtest gtest_main
    fledge::lib
  )
  add_test(${name} ${name}_cuda_exe)
  set_property(TARGET ${name}_cuda_exe PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  # target_compile_features(${name}_cuda_exe PUBLIC cuda_std_20)
endmacro()

flg_add_cuda_test(rng_tests)

flg_add_test(vector_tests)
flg_add_test(accel_tests)
flg_add_test(shape_tests)
flg_add_test(embree_tests)
flg_add_test(distribution_tests)
flg_add_test(scene_tests)
flg_add_test(resource_tests)
flg_add_test(sampler_tests)
flg_add_test(dispatcher_tests)
