#ifdef SIMDPP_PREVIEW
#define SIMDPP_DISPATCH_NAMESPACE arch_preview
#define SIMDPP_ARCH_X86_AVX
#else
#define SIMDPP_DISPATCH_NAMESPACE SIMDPP_ARCH_NAMESPACE
#endif

#include <cpu/gemm.hpp>

#include <simdpp/simd.h>

#include <simdpp/dispatch/get_arch_raw_cpuid.h>

#define SIMDPP_USER_ARCH_INFO ::simdpp::get_arch_raw_cpuid()

namespace blas
{

namespace SIMDPP_DISPATCH_NAMESPACE
{

milliseconds
gemm(
    const std::execution::parallel_unsequenced_policy &,
    size_t N, size_t M, size_t P,
    float * result,
    float alpha, const float * a, const float * b,
    float beta, const float * c
)
{
    using vector_gemm_t = simdpp::float32<SIMDPP_FAST_FLOAT32_SIZE>;
    constexpr size_t vectorSize = sizeof(vector_gemm_t) / sizeof(float);
    constexpr size_t blockSize = vectorSize * 8;
    constexpr size_t blockVectors = blockSize / vectorSize;

    static_assert(blockSize % vectorSize == 0);

    const auto start = std::chrono::high_resolution_clock::now();

    size_t rowBlocks = (N + blockSize - 1) / blockSize;
    size_t columnBlocks = (P + blockSize - 1) / blockSize;

    size_t internalBlocks = (M + blockSize - 1) / blockSize;

    #pragma omp parallel
    {
        alignas(max_align_t) float localA[blockSize][blockSize];
        alignas(max_align_t) float localB[blockSize][blockSize];
        alignas(max_align_t) float localResult[blockSize][blockSize][vectorSize];

        #pragma omp for
        for (size_t blockRow = 0; blockRow < rowBlocks; ++blockRow) {
            for (size_t blockColumn = 0; blockColumn < columnBlocks; ++blockColumn) {
                const size_t currentBlockRowCount = std::min(N - blockRow * blockSize, blockSize);
                const size_t currentBlockColumnCount = std::min(P - blockColumn * blockSize, blockSize);

                // zero out localResult
                std::memset(localResult, 0, blockSize * blockSize * vectorSize * sizeof(float));

                for (size_t blockInternal = 0; blockInternal < internalBlocks; ++blockInternal) {
                    const size_t currentBlockInternalCount = std::min(M - blockInternal * blockSize, blockSize);

                    // copy to localA
                    std::memset(localA, 0, blockSize * blockSize * sizeof(float));
                    for (size_t rowA = 0; rowA < currentBlockRowCount; ++rowA) {
                        for (size_t columnA = 0; columnA < currentBlockInternalCount; ++columnA) {
                            const size_t fullRowA = (blockRow * blockSize + rowA);
                            const size_t fullColumnA = (blockInternal * blockSize + columnA);
                            const size_t aIndex = fullRowA * M + fullColumnA;

                            localA[rowA][columnA] = a[aIndex];
                        }
                    }
                    // copy to localB
                    std::memset(localB, 0, blockSize * blockSize * sizeof(float));
                    for (size_t rowB = 0; rowB < currentBlockInternalCount; ++rowB) {
                        for (size_t columnB = 0; columnB < currentBlockColumnCount; ++columnB) {
                            const size_t fullRowB = (blockInternal * blockSize + rowB);
                            const size_t fullColumnB = (blockColumn * blockSize + columnB);
                            const size_t bIndex = fullRowB * P + fullColumnB;

                            localB[columnB][rowB] = b[bIndex];
                        }
                    }

                    // accumulate GEMM in localResult
                    for (size_t row = 0; row < currentBlockRowCount; ++row) {
                        for (size_t column = 0; column < currentBlockColumnCount; ++column) {
                            const float zero = 0.0f;

                            auto vectorSum = simdpp::load_splat<vector_gemm_t>(&zero);
                            for (size_t internal = 0; internal < blockVectors; ++internal) {
                                const size_t offset = internal * vectorSize;
                                vectorSum = simdpp::fmadd(
                                    simdpp::load_u<vector_gemm_t>(&localA[row][offset]),
                                    simdpp::load_u<vector_gemm_t>(&localB[column][offset]),
                                    vectorSum
                                );
                            }

                            simdpp::store_u(localResult[row][column], simdpp::add(simdpp::load_u<vector_gemm_t>(localResult[row][column]), vectorSum));
                        }
                    }
                }

                // copy localResult to result
                for (size_t rowResult = 0; rowResult < currentBlockRowCount; ++rowResult) {
                    for (size_t columnResult = 0; columnResult < currentBlockColumnCount; ++columnResult) {
                        size_t fullRowResult = (blockRow * blockSize + rowResult);
                        size_t fullColumnResult = (blockColumn * blockSize + columnResult);
                        size_t resultIndex = fullRowResult * P + fullColumnResult;

                        result[resultIndex] = simdpp::reduce_add(simdpp::load_u<vector_gemm_t>(localResult[rowResult][columnResult]));
                    }
                }
            }
        }

        #pragma omp barrier

        const size_t columnVectorCount = P / vectorSize;
        const size_t columnLeftoverOffset = columnVectorCount * vectorSize;

        const auto vectorAlpha = simdpp::load_splat<vector_gemm_t>(&alpha);
        const auto vectorBeta = simdpp::load_splat<vector_gemm_t>(&beta);

        #pragma omp for
        for (size_t row = 0; row < N; ++row) {
            const size_t rowOffset = row * P;
            //vectorized
            for (size_t column = 0; column < columnVectorCount; ++column) {
                const auto offset = rowOffset + column * vectorSize;

                simdpp::store(
                    &result[offset],
                    simdpp::fmadd(
                        vectorAlpha,
                        simdpp::load_u<vector_gemm_t>(&result[offset]),
                        simdpp::mul(
                            vectorBeta,
                            simdpp::load_u<vector_gemm_t>(&c[offset])
                        )
                    )
                );
            }
            //leftover
            for (size_t column = columnLeftoverOffset; column < P; ++column) {
                size_t index = row * P + column;
                result[index] = alpha * result[index] + beta * c[index];
            }
        }
    }

    const auto stop = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<milliseconds>(stop - start);
}

}

SIMDPP_MAKE_DISPATCHER((milliseconds) (gemm)(
    (const std::execution::parallel_unsequenced_policy &) expo,
    (size_t) N, (size_t) M, (size_t) P,
    (float *) result,
    (float) alpha, (const float *) a, (const float *) b,
    (float) beta, (const float *) c
)
);

}