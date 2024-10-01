#pragma once

#include <cpu/common.hpp>

namespace blas
{

milliseconds
gemm(const std::execution::sequenced_policy &,
     size_t N, size_t M, size_t P,
     float * result,
     float alpha, const float * a, const float * b,
     float beta, const float * c);

milliseconds
gemm(const std::execution::parallel_policy &,
     size_t N, size_t M, size_t P,
     float * result,
     float alpha, const float * a, const float * b,
     float beta, const float * c);

milliseconds
gemm(const std::execution::parallel_unsequenced_policy &,
     size_t N, size_t M, size_t P,
     float * result,
     float alpha, const float * a, const float * b,
     float beta, const float * c);

namespace arch_preview
{

milliseconds
gemm(const std::execution::parallel_unsequenced_policy &,
     size_t N, size_t M, size_t P,
     float * result,
     float alpha, const float * a, const float * b,
     float beta, const float * c);

}

}
