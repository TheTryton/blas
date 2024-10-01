cmake_minimum_required(VERSION 3.30)

include(SimdppMultiarch)

set(OpenMP_RUNTIME_MSVC "llvm")
find_package(OpenMP REQUIRED)

find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/dispatch" blas_cpu_dispatch_files)
simdpp_get_compilable_archs(runnable_archs)
set(blas_dispatch_sources "")
foreach (dispatch_file ${blas_cpu_dispatch_files})
    simdpp_multiarch(dispatch_file_sources ${dispatch_file} ${runnable_archs})
    list(APPEND blas_dispatch_sources ${dispatch_file_sources})
endforeach ()

add_library(blas_cpu_dispatch STATIC ${blas_dispatch_sources})
target_include_directories(blas_cpu_dispatch PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(blas_cpu_dispatch PRIVATE ${SIMDPP_INCLUDE_DIRS})
target_link_libraries(blas_cpu_dispatch PRIVATE blas_core ${OpenMP_LIBRARIES})
target_compile_options(blas_cpu_dispatch PRIVATE ${OpenMP_CXX_FLAGS})
set_target_properties(blas_cpu_dispatch
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

set(MKL_LINK "static")
find_package(MKL CONFIG REQUIRED)

find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/src" blas_cpu_sources)

add_library(blas_cpu STATIC ${blas_cpu_dispatch_files} ${blas_cpu_sources})
target_include_directories(blas_cpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(blas_cpu PRIVATE ${SIMDPP_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(blas_cpu PRIVATE blas_core blas_cpu_dispatch MKL::MKL ${OpenMP_LIBRARIES})
target_compile_options(blas_cpu PRIVATE ${OpenMP_CXX_FLAGS})
target_compile_definitions(blas_cpu PRIVATE SIMDPP_PREVIEW)
set_target_properties(blas_cpu
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)