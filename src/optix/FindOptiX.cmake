if(DEFINED ENV{OptiX_INSTALL_DIR})
  message(STATUS "Detected OptiX_INSTALL_DIR in environment variable: ${OptiX_INSTALL_DIR}")
  find_path(OptiX_ROOT_DIR NAMES include/optix.h PATHS $ENV{OptiX_INSTALL_DIR})
elseif(WIN32)
  find_path(OptiX_ROOT_DIR
    NAMES include/optix.h
    PATHS
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.5.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.2.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.1.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0"
  )
  message(STATUS "Searched OptiX_ROOT_DIR: ${OptiX_ROOT_DIR}")
else()
  find_path(OptiX_ROOT_DIR NAMES include/optix.h)
  message(STATUS "Searched OptiX_ROOT_DIR: ${OptiX_ROOT_DIR}")
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX
  FAIL_MESSAGE "Failed to find OptiX install dir. Please instal OptiX or set OptiX_INSTALL_DIR environment variable."
  REQUIRED_VARS OptiX_ROOT_DIR)

add_library(OptiX INTERFACE IMPORTED)
target_include_directories(OptiX INTERFACE "${OptiX_ROOT_DIR}/include")

find_path(OptiX_INCLUDE
  NAMES optix.h
  PATHS "${OptiX_ROOT_DIR}/include"
  NO_DEFAULT_PATH
)