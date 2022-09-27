# # OIDN is locally installed
CPMAddPackage(
  URL https://github.com/OpenImageDenoise/oidn/releases/download/v1.4.3/oidn-1.4.3.src.tar.gz
  NAME OpenImageDenoise
  GIT_TAG v1.4.3
  VERSION 1.4.3
  OPTIONS "OIDN_APPS OFF" "OIDN_API_NAMESPACE ON"
)

add_library(fledge_lib_oidn oidn.cpp)
target_include_directories(fledge_lib_oidn PUBLIC ${CMAKE_SOURCE_DIR}/src/)
target_link_libraries(fledge_lib_oidn PUBLIC OpenImageDenoise)

add_library(fledge::lib::oidn ALIAS fledge_lib_oidn)
