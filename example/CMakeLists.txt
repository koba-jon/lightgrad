cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(Example CXX)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../lightgrad)

# Find Package
find_package(LightGrad REQUIRED)

# Directory Name
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Set Compiler Options
set(CMAKE_CXX_FLAGS "-O2 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -fsanitize=address -fno-omit-frame-pointer -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall")

# Set Source Code
set(SRCS
    main.cpp
)

# Link
add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME} lightgrad)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

