cmake_minimum_required(VERSION 3.0.0)
project(Tensor)

option(ENABLE_CUDA "build support for computation on NVidia GPUs (CUDA also needs to be installed)" OFF)
option(PERLIN_MODULE "build PerlinNoise lib" ON)

if (NOT ${CMAKE_SIZEOF_VOID_P} EQUAL "8" AND ENABLE_CUDA)
    message(FATAL_ERROR "only x64 architecture is avaliable")
endif()

set(CMAKE_CXX_STANDARD 17)

# place all predefined target projects into a filter
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(PREDEFINED_TARGETS_FOLDER "CustomTargets")

add_subdirectory(${PROJECT_SOURCE_DIR}/Tensor Tensor)

if(ENABLE_CUDA)
    add_subdirectory(${PROJECT_SOURCE_DIR}/TensorCuda TensorCuda)
endif()

if(PERLIN_MODULE)
    add_subdirectory(${PROJECT_SOURCE_DIR}/modules/PerlinNoise)
endif()
