#include <cpu/gemv.hpp>

#include <mkl.h>

#include <algorithm>
#include <thread>

namespace blas
{

milliseconds
gemv(
    const std::execution::parallel_policy &,
    size_t N, size_t M,
    float * result,
    float alpha, const float * a, const float * x,
    float beta, const float * y
)
{
    std::copy(y, y + N, result);

    mkl_set_num_threads(std::thread::hardware_concurrency());

    const auto start = std::chrono::high_resolution_clock::now();
    cblas_sgemv(CblasRowMajor, CblasNoTrans, N, M, alpha, a, M, x, 1, beta, result, 1);
    const auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<milliseconds>(stop - start);
}

}