project(TensorCuda LANGUAGES CXX CUDA)

set(INCLUDE_FILES
    include/TensorCuda.cuh 
    include/TensorCuda.ipp 
    include/TensorOperatorKernels.cuh
)

add_library(${PROJECT_NAME}
    src/TensorCuda.cu

    ${INCLUDE_FILES}
)

source_group("Header Files" FILES ${INCLUDE_FILES})

add_compile_definitions(_CUDA)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)

target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_directories(${PROJECT_NAME} PRIVATE include)

target_link_libraries(${PROJECT_NAME} Tensor)

