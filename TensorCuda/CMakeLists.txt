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

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    OUTPUT_NAME ${PROJECT_NAME}_$<CONFIG>
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin/
    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/bin/
    CUDA_STANDARD 17
)

target_compile_definitions(${PROJECT_NAME} PUBLIC _TS_CUDA)

source_group("Header Files" FILES ${INCLUDE_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(MSVC)
target_compile_options(${PROJECT_NAME} PRIVATE -Xcompiler="/Zc:__cplusplus")
endif()

if(ENABLE_OPENMP)
    target_compile_options(${PROJECT_NAME} PRIVATE -Xcompiler="/openmp")
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PUBLIC ./)

target_link_directories(${PROJECT_NAME} PRIVATE include)

target_link_libraries(${PROJECT_NAME} Tensor)

