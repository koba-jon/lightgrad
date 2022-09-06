cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(liblightgrad CXX)

# Directory Name
set(LightGrad_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LightGrad_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../lightgrad)

# Add library
add_library(lightgrad
    SHARED
    ${LightGrad_SRC_DIR}/tensor.cpp
    ${LightGrad_SRC_DIR}/operator.cpp
    ${LightGrad_SRC_DIR}/functional.cpp
    ${LightGrad_SRC_DIR}/optimizer.cpp
)

set(LightGrad_HEADERS
    ${LightGrad_INCLUDE_DIR}/lightgrad/lightgrad.hpp
    ${LightGrad_INCLUDE_DIR}/lightgrad/declare.hpp
    ${LightGrad_INCLUDE_DIR}/lightgrad/tensor.hpp
    ${LightGrad_INCLUDE_DIR}/lightgrad/operator.hpp
    ${LightGrad_INCLUDE_DIR}/lightgrad/functional.hpp
    ${LightGrad_INCLUDE_DIR}/lightgrad/optimizer.hpp
)

set_target_properties(lightgrad PROPERTIES PUBLIC_HEADER "${LightGrad_HEADERS}")
include_directories(${LightGrad_INCLUDE_DIR})

install(TARGETS lightgrad
    EXPORT lightgrad-export
    LIBRARY DESTINATION lib
    INCLUDES DESTINATION include
    PUBLIC_HEADER DESTINATION include/lightgrad
)

install(EXPORT lightgrad-export
    FILE lightgrad-config.cmake
    DESTINATION share/cmake/lightgrad/
    EXPORT_LINK_INTERFACE_LIBRARIES
)

