enable_language(ISPC)

add_library(fledge_lib_ispc STATIC vector.ispc)
target_include_directories(fledge_lib_ispc PUBLIC ${CMAKE_SOURCE_DIR})
set_target_properties(
    fledge_lib_ispc
    PROPERTIES
    LINKER_LANGUAGE C
    ISPC_INSTRUCTION_SETS avx2
)

add_library(fledge::lib::ispc ALIAS fledge_lib_ispc)
