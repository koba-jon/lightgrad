cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(Example)

# Directory Name
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Set Compiler Options
set(CMAKE_CXX_FLAGS "-O2 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -fsanitize=address -fno-omit-frame-pointer -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall")

# Set Source Code
set(SRCS
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/tensor.cpp
    ${SRC_DIR}/operator.cpp
    ${SRC_DIR}/functional.cpp
    ${SRC_DIR}/optimizer.cpp
)

# Link
add_executable(${PROJECT_NAME} ${SRCS})
include_directories(${INCLUDE_DIR})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
