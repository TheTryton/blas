#pragma once

#include <gpu/common.hpp>

namespace blas
{
milliseconds
gemv(const execution::parallel_gpu::cublas &,
     size_t N, size_t M,
     float * result,
     float alpha, const float * a, const float * x,
     float beta, const float * y);
}
