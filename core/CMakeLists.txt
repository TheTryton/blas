cmake_minimum_required(VERSION 3.30)

find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/include" blas_core_headers)
find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/src" blas_core_sources)

add_library(blas_core INTERFACE)
target_include_directories(blas_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
set_target_properties(blas_core
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)