#pragma once

#include <gpu/common.hpp>

namespace blas
{
milliseconds
gemv(const std::execution::parallel_gpu &,
     size_t N, size_t M,
     float * result,
     float alpha, const float * a, const float * x,
     float beta, const float * y);
}
