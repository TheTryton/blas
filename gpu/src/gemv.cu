#include <gpu/gemv.hpp>

#include <src_common.hpp>

namespace blas
{

milliseconds
gemv(const std::execution::parallel_gpu &,
     size_t N, size_t M,
     float * result,
     float alpha, const float * a, const float * x,
     float beta, const float * y)
{
    cudaStream_t stream;
    cudaAssert(cudaStreamCreate(&stream));

    cublasHandle_t handle;
    cublasAssert(cublasCreate_v2(&handle));

    float * aDevice;
    float * xDevice;
    float * yDevice;
    cudaAssert(cudaMalloc(&aDevice, N * M * sizeof(float)));
    cudaAssert(cudaMalloc(&xDevice, M * sizeof(float)));
    cudaAssert(cudaMalloc(&yDevice, N * sizeof(float)));

    cudaEvent_t startGpu;
    cudaEvent_t stopGpu;
    cudaAssert(cudaEventCreate(&startGpu));
    cudaAssert(cudaEventCreate(&stopGpu));

    cublasAssert(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED));
    cublasAssert(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    cublasAssert(cublasSetStream(handle, stream));

    cublasAssert(cublasSetMatrixAsync(N, M, sizeof(float), a, M, aDevice, M, stream));
    cublasAssert(cublasSetVectorAsync(M, sizeof(float), x, 1, xDevice, 1, stream));
    cublasAssert(cublasSetVectorAsync(N, sizeof(float), y, 1, yDevice, 1, stream));

    cublasAssert(cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha, aDevice, M, xDevice, 1, &beta, yDevice, 1));
    cudaAssert(cudaEventRecord(startGpu, stream));
    cublasAssert(cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha, aDevice, M, xDevice, 1, &beta, yDevice, 1));
    cudaAssert(cudaEventRecord(stopGpu, stream));

    cublasAssert(cublasGetVectorAsync(N, sizeof(float), yDevice, 1, result, 1, stream));

    cudaAssert(cudaStreamSynchronize(stream));

    float gpuMs;
    cudaAssert(cudaEventElapsedTime(&gpuMs, startGpu, stopGpu));

    cudaAssert(cudaDeviceSynchronize());

    cudaAssert(cudaEventDestroy(stopGpu));
    cudaAssert(cudaEventDestroy(startGpu));

    cudaAssert(cudaFree(aDevice));
    cudaAssert(cudaFree(xDevice));
    cudaAssert(cudaFree(yDevice));

    cublasAssert(cublasDestroy(handle));

    cudaAssert(cudaStreamDestroy(stream));

    return milliseconds{gpuMs};
}

}