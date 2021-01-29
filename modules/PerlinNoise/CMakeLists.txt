project(PerlinNoise)

set(INCLUDE_FILES
    include/TensorPerlin.h
    include/TensorPerlin.ipp
)

add_executable(${PROJECT_NAME}
    src/TensorPerlin.cpp

    ${INCLUDE_FILES}
)

source_group("Header Files" FILES ${INCLUDE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_libraries(${PROJECT_NAME} Tensor)
