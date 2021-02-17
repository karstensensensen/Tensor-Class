project(Tensor)

set(INCLUDE_FILES
    include/SaveSequence.h
    include/SaveSequence.ipp
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

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    OUTPUT_NAME ${PROJECT_NAME}_$<CONFIG>
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin/
    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/bin/
)

target_compile_definitions(${PROJECT_NAME} PUBLIC $<$<CONFIG:DEBUG>:_TS_DEBUG>)

if(ENABLE_OPENMP)
    target_compile_options(${PROJECT_NAME} PRIVATE /openmp)
endif()

source_group("Header Files" FILES ${INCLUDE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC include)
