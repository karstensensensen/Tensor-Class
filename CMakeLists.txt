cmake_minimum_required(VERSION 3.0.0)
project(Tensor)

option(ENABLE_CUDA      "build support for computation on NVidia GPUs (CUDA also needs to be installed)" OFF)
#option(PERLIN_MODULE "build PerlinNoise lib" ON)
option(ENABLE_OPENMP    "add /openmp flag to compiler to enable multithreaded computing" OFF)
option(STATIC_MODULE    "build a static version that might be faster, but not able to be resized (work in progress)" OFF)
option(OPENCV_MODULE    "build a static library with helpful functions for OpenCV (like converting a tensor to an image)" OFF)

if (NOT ${CMAKE_SIZEOF_VOID_P} EQUAL "8" AND ENABLE_CUDA)
    message(FATAL_ERROR "only x64 architecture is avaliable")
endif()

set(CMAKE_CXX_STANDARD 20)

# place all predefined target projects into a filter
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(PREDEFINED_TARGETS_FOLDER "CustomTargets")

add_subdirectory(${PROJECT_SOURCE_DIR}/Tensor Tensor)

if(OPENCV_MODULE)
    add_subdirectory(${PROJECT_SOURCE_DIR}/modules/TensorOpenCV TensorOpenCV)
endif()

if(ENABLE_CUDA)
    add_subdirectory(${PROJECT_SOURCE_DIR}/TensorCuda TensorCuda)
endif()

# if(PERLIN_MODULE)
#     add_subdirectory(${PROJECT_SOURCE_DIR}/modules/PerlinNoise)
# endif()

if(STATIC_MODULE)
    add_subdirectory(${PROJECT_SOURCE_DIR}/StaticTensor StaticTensor)
endif()

if(OPENCV_MODULE)
    
endif()

