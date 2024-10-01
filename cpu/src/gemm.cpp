#include <cpu/gemm.hpp>

#include <mkl.h>

#include <algorithm>
#include <thread>

namespace blas
{

milliseconds
gemm(const std::execution::sequenced_policy &,
     size_t N, size_t M, size_t P,
     float * result,
     float alpha, const float * a, const float * b,
     float beta, const float * c)
{
    const auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (size_t row = 0; row < N; ++row) {
        for (size_t column = 0; column < P; ++column) {
            float sum = 0.0f;
            float compensation = 0.0f;
            for (size_t internal = 0; internal < M; ++internal) {
                float value = a[row * M + internal] * b[internal * P + column];
                float compensatedValue = value - compensation;
                float newSum = sum + compensatedValue;
                compensation = (newSum - sum) - value;
                sum = newSum;
            }
            result[row * P + column] = sum;
        }
    }

    for (size_t row = 0; row < N; ++row) {
        for (size_t column = 0; column < P; ++column) {
            result[row * P + column] = alpha * result[row * P + column] + beta * c[row * P + column];
        }
    }

    const auto stop = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<milliseconds>(stop - start);
}

milliseconds
gemm(
    const std::execution::parallel_policy &,
    size_t N, size_t M, size_t P,
    float * result,
    float alpha, const float * a, const float * b,
    float beta, const float * c
)
{
    std::copy(c, c + N * P, result);

    mkl_set_num_threads(std::thread::hardware_concurrency());

    const auto start = std::chrono::high_resolution_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, P, M, alpha, a, M, b, P, beta, result, P);
    const auto stop = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<milliseconds>(stop - start);
}

}

/*
inline float reduceAdd(__m256 v)
{
    v = _mm256_hadd_ps(_mm256_permute2f128_ps(v, v, 0x20), _mm256_permute2f128_ps(v, v, 0x31));
    return _mm256_cvtss_f32(v);
}

inline __m256 load(const float * vp)
{
    return _mm256_loadu_ps(vp);
}

inline void store(float * vp, __m256 v)
{
    _mm256_storeu_ps(vp, v);
}

inline __m256 set(float v)
{
    return _mm256_broadcast_ss(&v);
}

inline __m256 add(__m256 a, __m256 b)
{
    return _mm256_add_ps(a, b);
}

inline __m256 mul(__m256 a, __m256 b)
{
    return _mm256_mul_ps(a, b);
}

milliseconds
gemm(
    const std::execution::parallel_unsequenced_policy &,
    size_t N, size_t M, size_t P,
    float *result,
    float alpha, const float *a, const float *b,
    float beta, const float *c
)
{
    constexpr size_t blockSize = std::hardware_destructive_interference_size * 4;
    using vector_t = __m256;
    constexpr size_t vectorSize = sizeof(__m256);
    constexpr size_t blockVectorCount = blockSize / vectorSize;

    static_assert(blockSize % vectorSize == 0);

    const auto start = std::chrono::high_resolution_clock::now();

    size_t rowBlocks = (N + blockSize - 1) / blockSize;
    size_t columnBlocks = (P + blockSize - 1) / blockSize;

    size_t internalBlocks = (M + blockSize - 1) / blockSize;

    #pragma omp parallel
    {
        alignas(max_align_t) float localA[blockSize][blockSize];
        alignas(max_align_t) float localB[blockSize][blockSize];
        alignas(max_align_t) float localResult[blockSize][blockSize];

        #pragma omp for
        for (size_t blockRow = 0; blockRow < rowBlocks; ++blockRow) {
            for (size_t blockColumn = 0; blockColumn < columnBlocks; ++blockColumn) {
                const size_t currentBlockRowCount = std::min(N - blockRow * blockSize, blockSize);
                const size_t currentBlockColumnCount = std::min(P - blockColumn * blockSize, blockSize);

                // zero out localResult
                for (size_t rowResult = 0; rowResult < blockSize ++rowResult) {
                    for (size_t columnResult = 0; columnResult < blockSize; ++columnResult) {
                        localResult[rowResult][columnResult] = 0.0f;
                    }
                }

                for (size_t blockInternal = 0; blockInternal < internalBlocks; ++blockInternal) {
                    const size_t currentBlockInternalCount = std::min(M - blockInternal * blockSize, blockSize);

                    // copy to localA
                    for (size_t rowA = 0; rowA < currentBlockRowCount; ++rowA) {
                        for (size_t columnA = 0; columnA < currentBlockInternalCount; ++columnA) {
                            const size_t fullRowA = (blockRow * blockSize + rowA);
                            const size_t fullColumnA = (blockInternal * blockSize + columnA);
                            const size_t aIndex = fullRowA * M + fullColumnA;

                            localA[rowA][columnA] = a[aIndex];
                        }
                        for (size_t columnA = currentBlockInternalCount; columnA < blockSize; ++columnA) {
                            localA[rowA][columnA] = 0.0f;
                        }
                    }
                    // copy to localB
                    for (size_t rowB = 0; rowB < currentBlockInternalCount; ++rowB) {
                        for (size_t columnB = 0; columnB < currentBlockColumnCount; ++columnB) {
                            const size_t fullRowB = (blockInternal * blockSize + rowB);
                            const size_t fullColumnB = (blockColumn * blockSize + columnB);
                            const size_t bIndexTranspose = fullColumnB * P + fullRowB;

                            localB[rowB][columnB] = a[bIndexTranspose];
                        }
                        for (size_t columnB = currentBlockColumnCount; columnB < blockSize; ++columnB) {
                            localB[rowB][columnB] = 0.0f;
                        }
                    }

                    const size_t currentBlockInternalVectorCount = currentBlockInternalCount / vectorSize;
                    const size_t currentBlockInternalLeftoverOffset = currentBlockInternalVectorCount * vectorSize;
                    const size_t currentBlockInternalLeftoverCount = currentBlockInternalCount - currentBlockInternalVectorCount;

                    // accumulate GEMM in localResult
                    for (size_t row = 0; row < currentBlockRowCount; ++row) {
                        for (size_t column = 0; column < currentBlockColumnCount; ++column) {
                            vector_t localResultVectorAccum = set(0.0);
                            for (size_t internal = 0; internal < blockVectorCount; ++internal) {
                                const auto offset = internal * vectorSize;
                                const auto vectorA = load(localA[row] + offset);
                                const auto vectorB = load(localB[row] + offset);

                                localResultVectorAccum = add(localResultVectorAccum, mul(vectorA, vectorB));
                            }
                            float localResultAccum = reduceAdd(localResultVectorAccum);
                            localResult[row][column] += localResultAccum;
                        }
                    }
                }

                // copy localResult to result
                for (size_t rowResult = 0; rowResult < currentBlockRowCount; ++rowResult) {
                    for (size_t columnResult = 0; columnResult < currentBlockColumnCount; ++columnResult) {
                        size_t fullRowResult = (blockRow * blockSize + rowResult);
                        size_t fullColumnResult = (blockColumn * blockSize + columnResult);
                        size_t resultIndex = fullRowResult * P + fullColumnResult;

                        result[resultIndex] = localResult[rowResult][columnResult];
                    }
                }
            }
        }
    }

    const size_t columnVectorCount = P / vectorSize;
    const size_t columnLeftoverOffset = columnVectorCount * vectorSize;
    const size_t columnLeftoverCount = P - columnVectorCount;

    #pragma omp parallel
    {
        const auto vectorAlpha = set(alpha);
        const auto vectorBeta = set(beta);

        //vectorized
        #pragma omp for
        for (size_t row = 0; row < N; ++row) {
            const size_t rowOffset = row * P;
            for (size_t column = 0; column < columnVectorCount; ++column) {
                const auto offset = rowOffset + column * vectorSize;
                const auto vectorResult = load(result + offset);
                const auto vectorC = load(c + offset);

                const auto vectorNewResult = _mm256_fmadd_ps(vectorAlpha, vectorResult, _mm256_add_ps(vectorBeta, vectorC));

                store(result + offset, vectorNewResult);
            }
        }

        //leftover
        #pragma omp for
        for (size_t row = 0; row < N; ++row) {
            for (size_t column = columnLeftoverOffset; column < columnLeftoverCount; ++column) {
                size_t index = row * P + column;
                result[index] = alpha * result[index] + beta * c[index];
            }
        }
    }

    const auto stop = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<milliseconds>(stop - start);
}
*/
