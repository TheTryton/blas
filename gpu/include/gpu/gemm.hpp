#pragma once

#include <gpu/common.hpp>

namespace blas
{
milliseconds
gemm(const execution::parallel_gpu::own &,
     size_t N, size_t M, size_t P,
     float * result,
     float alpha, const float * a, const float * b,
     float beta, const float * c);

milliseconds
gemm(const execution::parallel_gpu::cublas &,
     size_t N, size_t M, size_t P,
     float * result,
     float alpha, const float * a, const float * b,
     float beta, const float * c);

milliseconds
gemm(const execution::parallel_gpu::cublas_lt &,
     size_t N, size_t M, size_t P,
     float * result,
     float alpha, const float * a, const float * b,
     float beta, const float * c);
}
