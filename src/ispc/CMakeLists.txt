add_library(smallvol_lib_ispc STATIC vector.ispc)
target_include_directories(smallvol_lib_ispc PUBLIC ../)
set_target_properties(
    smallvol_lib_ispc
    PROPERTIES
    LINKER_LANGUAGE C 
    ISPC_INSTRUCTION_SETS avx2
)