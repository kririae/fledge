# The material implementations are seperated because material evaluation is extensiable
add_library(fledge_materials builtin_materials.cpp material_base.cpp)
target_include_directories(fledge_materials PUBLIC ${CMAKE_SOURCE_DIR}/src/)
