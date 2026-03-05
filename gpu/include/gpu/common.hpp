#pragma once

#include <core/blas.hpp>

#include <execution>

namespace execution
{

struct parallel_gpu
{
    struct cublas {};
    struct cublas_lt {};
    struct own {};
};

inline constexpr parallel_gpu par_gpu{};

}
