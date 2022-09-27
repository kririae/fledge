CPMAddPackage(
  NAME ftxui
  GITHUB_REPOSITORY ArthurSonzogni/FTXUI
  GIT_TAG v3.0.0
  OPTIONS "FTXUI_BUILD_EXAMPLES OFF" "FTXUI_BUILD_DOCS OFF"
)

add_library(fledge_lib_fxtui tui.cpp)
target_link_libraries(fledge_lib_fxtui PUBLIC ftxui::screen ftxui::component)
target_include_directories(fledge_lib_fxtui PUBLIC ${CMAKE_SOURCE_DIR}/src/)

add_library(fledge::lib::fxtui ALIAS fledge_lib_fxtui)
