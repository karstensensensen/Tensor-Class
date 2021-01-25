project(StaticTensor)

set(INCLUDE_FILES
    include/StaticTensor.h
    include/StaticTensor.ipp
)

add_library(${PROJECT_NAME} STATIC
    src/StaticTensor.cpp

    ${INCLUDE_FILES}
)

target_compile_definitions(${PROJECT_NAME} PUBLIC $<$<CONFIG:DEBUG>:_TS_DEBUG>)

if(ENABLE_OPENMP)
    target_compile_options(${PROJECT_NAME} PUBLIC /openmp)
endif()

source_group("Header Files" FILES ${INCLUDE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC include)
