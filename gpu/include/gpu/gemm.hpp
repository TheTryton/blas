#pragma once

#include <gpu/common.hpp>

namespace blas
{
milliseconds
gemm(const std::execution::parallel_gpu &,
     size_t N, size_t M, size_t P,
     float * result,
     float alpha, const float * a, const float * b,
     float beta, const float * c);
}
