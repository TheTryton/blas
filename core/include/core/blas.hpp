#pragma once

#include <chrono>
#include <type_traits>

namespace blas
{
using milliseconds = std::chrono::duration<double, std::milli>;

template<typename BlasExecutor>
struct is_blas_executor : std::false_type {};

}
