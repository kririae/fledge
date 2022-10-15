enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

find_program(BIN2C bin2c
  DOC "Path to the cuda-sdk bin2c executable.")

# code from https://github.com/ingowald/optix7course/blob/master/common/gdt/cmake/configure_optix.cmake
macro(cuda_compile_and_embed output_var cuda_file)
  set(c_var_name ${output_var})
  cuda_compile_ptx(ptx_files ${cuda_file} OPTIONS --generate-line-info -use_fast_math --keep)
  list(GET ptx_files 0 ptx_file)
  set(embedded_file ${ptx_file}_embedded.c)
  add_custom_command(
    OUTPUT ${embedded_file}
    COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
    DEPENDS ${ptx_file}
    COMMENT "compiling (and embedding ptx from) ${cuda_file}"
  )
  set(${output_var} ${embedded_file})
endmacro()
