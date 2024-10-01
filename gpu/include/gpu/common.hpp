#pragma once

#include <core/blas.hpp>

#include <execution>

namespace std
{
namespace execution
{
class parallel_gpu
{
};

inline constexpr parallel_gpu par_gpu{};
}

template<>
struct is_execution_policy<execution::parallel_gpu>
{
    constexpr static bool value = true;
};

}
