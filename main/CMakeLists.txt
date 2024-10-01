cmake_minimum_required(VERSION 3.30)

find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/include" blas_headers)
find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/src" blas_sources)

add_executable(blas ${blas_headers} ${blas_sources})
target_include_directories(blas PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(blas PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(blas PRIVATE blas_core blas_cpu blas_gpu)
set_target_properties(blas
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)