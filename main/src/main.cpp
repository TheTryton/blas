#include <cpu/gemv.hpp>
#include <cpu/gemm.hpp>
#include <gpu/gemv.hpp>
#include <gpu/gemm.hpp>

#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
#include <span>
#include <iostream>
#include <iomanip>

using namespace std;

template<typename ElementType>
ElementType generateRandom()
{
    static random_device rd;
    static mt19937 generator(rd());
    static uniform_real_distribution<ElementType> distribution(static_cast<ElementType>(0), static_cast<ElementType>(1));

    return distribution(generator);
}

template<typename ElementType>
vector<ElementType> createVector(size_t N, bool zero)
{
    vector<ElementType> v(N, static_cast<ElementType>(0));
    if (!zero) {
        generate(begin(v), end(v), generateRandom<ElementType>);
        //iota(begin(v), end(v), static_cast<ElementType>(1));
    }
    return v;
}

template<typename ElementType>
vector<ElementType> createMatrix(size_t N, size_t M, bool zero)
{
    vector<ElementType> v(N*M, static_cast<ElementType>(0));
    if (!zero) {
        generate(begin(v), end(v), generateRandom<ElementType>);
        //iota(begin(v), end(v), static_cast<ElementType>(1));
    }
    return v;
}

template<typename ElementType>
void compareVectors(span<ElementType> a, span<ElementType> b)
{
    if(size(a) != size(b)) {
        cerr << "Vector sizes differ!" << endl;
    }

    vector<ElementType> differences(size(a));

    for (size_t i = 0; i < size(a); ++i)
    {
        const auto difference = std::abs(a[i] - b[i]);
        differences[i] = difference;
    }

    const auto maxDiff = *max_element(begin(differences), end(differences));
    const auto avgDiff = accumulate(begin(differences), end(differences), static_cast<ElementType>(0)) / size(a);

    cout << format("Maximum difference={}", maxDiff) << endl;
    cout << format("Average difference={}", avgDiff) << endl;
}

template<typename ElementType>
void compareMatrices(span<ElementType> a, span<ElementType> b, size_t N, size_t M)
{
    if(size(a) != size(b)) {
        cerr << "Matrix sizes differ!" << endl;
    }

    vector<ElementType> differences(N * M);

    for (size_t row = 0; row < N; ++row) {
        for (size_t column = 0; column < M; ++column) {
            const auto i = row * M + column;
            const auto difference = std::abs(a[i] - b[i]);
            differences[i] = difference;
        }
    }

    const auto maxDiff = *max_element(begin(differences), end(differences));
    const auto avgDiff = accumulate(begin(differences), end(differences), static_cast<ElementType>(0)) / size(a);

    cout << format("Maximum difference={}", maxDiff) << endl;
    cout << format("Average difference={}", avgDiff) << endl;
}

double calculateTFLOPS(size_t N, size_t M, size_t P, blas::milliseconds milliseconds)
{
    return N * M * (2 * P + 3) / (milliseconds.count() / 1000.0) / 1024 / 1024 / 1024 / 1024;
}

template<typename ElementType>
void printMatrix(span<ElementType> m, size_t N, size_t M)
{
    for (size_t row = 0; row < N; ++row) {
        for (size_t column = 0; column < M; ++column) {
            cout << setw(6) << m[row * M + column] << ' ';
        }
        cout << endl;
    }
}

int main()
{
    using elementType = float;
    constexpr size_t N = 2048;
    constexpr size_t M = 2048;
    constexpr size_t P = 2048;

    const auto alpha = static_cast<elementType>(1024.0f);
    const auto beta = static_cast<elementType>(1024.0f);

    const auto a = createMatrix<elementType>(N, M, false);
    const auto b = createMatrix<elementType>(M, P, false);
    const auto c = createMatrix<elementType>(N, P, false);
    auto resultCpuTrue = createMatrix<elementType>(N, P, true);
    auto resultCpuMine = createMatrix<elementType>(N, P, true);
    auto resultCpuMKL = createMatrix<elementType>(N, P, true);
    auto resultGpu = createMatrix<elementType>(N, P, true);

    const auto cpuTimeTrue = blas::gemm(std::execution::seq, N, M, P, data(resultCpuTrue), alpha, data(a), data(b), beta, data(c));
    const auto cpuTimeMine = blas::gemm(std::execution::par_unseq, N, M, P, data(resultCpuMine), alpha, data(a), data(b), beta, data(c));
    const auto cpuTimeMKL = blas::gemm(std::execution::par, N, M, P, data(resultCpuMKL), alpha, data(a), data(b), beta, data(c));
    const auto gpuTime = blas::gemm(std::execution::par_gpu, N, M, P, data(resultGpu), alpha, data(a), data(b), beta, data(c));

    compareMatrices<elementType>(resultCpuTrue, resultCpuMine, N, P);
    compareMatrices<elementType>(resultCpuTrue, resultCpuMKL, N, P);
    compareMatrices<elementType>(resultCpuTrue, resultGpu, N, P);

    //printMatrix<elementType>(resultCpuTrue, N, P);
    //printMatrix<elementType>(resultCpuMine, N, P);

    cout << format("Time CPU (true): {} ms, {} TFLOPS", cpuTimeTrue.count(), calculateTFLOPS(N, M, P, cpuTimeTrue)) << endl;
    cout << format("Time CPU (mine): {} ms, {} TFLOPS", cpuTimeMine.count(), calculateTFLOPS(N, M, P, cpuTimeMine)) << endl;
    cout << format("Time CPU (mkl): {} ms, {} TFLOPS", cpuTimeMKL.count(), calculateTFLOPS(N, M, P, cpuTimeMKL)) << endl;
    cout << format("Time GPU (cublas): {} ms, {} TFLOPS", gpuTime.count(), calculateTFLOPS(N, M, P, gpuTime)) << endl;

    return 0;
}

