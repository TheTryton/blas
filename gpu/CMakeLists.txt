cmake_minimum_required(VERSION 3.30)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 70)

find_package(CUDAToolkit REQUIRED)

find_cuda_files("${CMAKE_CURRENT_SOURCE_DIR}/src" blas_gpu_cuda_sources)
find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/include" blas_gpu_headers)
find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/src" blas_gpu_sources)

add_library(blas_gpu STATIC ${blas_gpu_cuda_sources} ${blas_gpu_headers} ${blas_gpu_sources})
target_include_directories(blas_gpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(blas_gpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(blas_gpu PRIVATE blas_core CUDA::cudart CUDA::cublas)
target_compile_options(blas_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_target_properties(blas_gpu
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)