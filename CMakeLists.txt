cmake_minimum_required(VERSION 3.30)
project(blas LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(common)

set(SIMDPP_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libsimdpp")
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/third_party/libsimdpp/cmake")

add_subdirectory(core)
add_subdirectory(cpu)
add_subdirectory(gpu)
add_subdirectory(main)
