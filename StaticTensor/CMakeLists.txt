project(StaticTensor)

set(INCLUDE_FILES
    include/StaticTensor.h
    include/StaticTensor.ipp
)

add_library(${PROJECT_NAME}
    src/StaticTensor.cpp

    ${INCLUDE_FILES}
)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    OUTPUT_NAME ${PROJECT_NAME}_$<CONFIG>
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEFINFO ${CMAKE_BINARY_DIR}/bin/
    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/bin/
)

if(ENABLE_OPENMP)
    target_compile_options(${PROJECT_NAME} PRIVATE /openmp)
endif()

target_compile_definitions(${PROJECT_NAME} PUBLIC $<$<CONFIG:DEBUG>:_TS_DEBUG>)

if(ENABLE_OPENMP)
    target_compile_options(${PROJECT_NAME} PUBLIC /openmp)
endif()

source_group("Header Files" FILES ${INCLUDE_FILES})

target_link_libraries(${PROJECT_NAME} Tensor)

target_include_directories(${PROJECT_NAME} PUBLIC include)
