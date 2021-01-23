project(Tensor)

set(INCLUDE_FILES
    include/PerlinKernel.cuh
    include/PerlinNoise.h
    include/Tensor.h
    include/Tensor.ipp
    include/TensorArithmetic.h
    include/TensorArithmeticOperators.ipp
    include/TensorCompareOperators.h
    include/TensorEnums.h
    include/TensorExceptions.h
    include/TensorMath.ipp
    include/TensorSlice.h
    include/TensorSlice.ipp
    include/TensorTools.h
    include/TensorTools.ipp
    include/TSliceWrapper.h
)

add_library(${PROJECT_NAME} STATIC
    src/Tensor.cpp
    src/TSliceWrapper.cpp

    ${INCLUDE_FILES}
)

target_compile_definitions(${PROJECT_NAME} PUBLIC $<$<CONFIG:DEBUG>:_TS_DEBUG>)

if(ENABLE_OPENMP)
    target_compile_options(${PROJECT_NAME} PUBLIC /openmp)
endif()

source_group("Header Files" FILES ${INCLUDE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC include)
