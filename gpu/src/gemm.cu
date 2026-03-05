#include <assert.h>
#include <gpu/gemm.hpp>
#include <cuda/barrier>

#include <src_common.hpp>

struct matrix
{
    float * ptr;
    size_t N;
    size_t M;

    __device__ __inline__ float& operator()(size_t row, size_t column) const
    {
        return ptr[row * M + column];
    }

    __device__ __inline__ float * end() const
    {
        return ptr + M * N;
    }

    __device__ __inline__ void advance(size_t rows, size_t columns)
    {
        ptr += rows * M + columns;
    }
};

struct const_matrix
{
    const float * ptr;
    size_t N;
    size_t M;

    __device__ inline const float& operator()(size_t row, size_t column) const
    {
        return ptr[row * M + column];
    }

    __device__ __inline__ const float * end() const
    {
        return ptr + M * N;
    }

    __device__ __inline__ void advance(size_t rows, size_t columns)
    {
        ptr += rows * M  + columns;
    }
};

__global__ void gemm_kernel(size_t N, size_t M, size_t P,
    float alpha, const float * a, const float * b,
    float beta, float * c)
{
    // Prereqs:
    // a, b, c are row-major
    // N, M, P are divisible by 4
    // a, b, c are aligned to alignof(float4) boundaries
    // There are several optimizations made in order to make the kernel run ~ cublaslt version,
    // from simplest ones to complex ones:

    // (*) 1. We want to coalesce accesses to matrix A and C
    // though it is not clearly visible in the code it still matters (see (*) markers)
    // this is mainly done through accessing columns by .x and rows by .y component
    // of threadIdx.x.
    // IMPORTANT: as coalescing is a warp level construct in CUDA
    // best way to utilize memory coalescing is to set the blockDim.x == warpSize (32)
    // and allocate rest of available blockSize to blockDim.y (almost always also 32)
    // IMPORTANT 2:
    // blockDim.x == warpSize - warp cannot wrap around "rows", otherwise we
    // don't coalesce properly

    // (**) 2. Utilize tiling and faster shared memory: tileSize = (warpSize, warpSize)
    // a) dot-product loop requires us to access memory in coalesced way for matrix A
    // but in a non coalesced way for matrix B from global memory - thus we coalesce global
    // memory accesses in tiles by copying data from global memory once per thread.
    // After that copy we must sync the threads in the block (__syncthreads())
    // b) we perform regular dot-product loop on shared memory which can be
    // "n-way"-coalesced for tile of B matrix

    // (***) 3. Push coalescing to it's limits and increase arithmetic intensity:
    // a) we can improve memory access pattern (+ byproduct increase arithmetic intensity)
    // by calculating results for 4 columns in each thread. First it allows us to load biggest
    // coalescable data type (float4) from matrix B (both from global and shared memory) and additionally
    // perform more computing in each thread - storing results in C matrix is also improved as we
    // can store 4 floats at once (float4)
    // b) we can additionally improve arithmetic intensity by also loading 4 rows of matrix A
    // thus calculating 16 C matrix entries - even though this loads global memory from far away regions
    // those regions are still coalesced between threads in a block
    // Note: last optimization pushes sharedMemory amount to 32KB which nearing the limit for a thread block

    extern __shared__ float sharedMemory[];

    if (blockDim.x != blockDim.y)
        return;

    size_t globalColumn = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // (*)
    size_t globalRow = (blockIdx.y * blockDim.y + threadIdx.y) * 4; // (*)

    if (globalRow >= N || globalColumn >= P)
        return;

    const_matrix A{a, N, M};
    const_matrix B{b, M, P};
    matrix C{c, N, P};

    matrix ATile{sharedMemory, blockDim.y * 4, blockDim.x};
    matrix BTile{ATile.end(), blockDim.y, blockDim.x * 4};

    size_t tileColumn = threadIdx.x;
    size_t tileRow = threadIdx.y;

    size_t globalRowBase =
        blockIdx.y * blockDim.y * 4 + tileRow * 4;

    size_t globalColumnBase =
        blockIdx.x * blockDim.x * 4 + tileColumn * 4;

    // (**), (***)
    float4 accumulatedValues[4] = {
        make_float4(0.0f, 0.0f, 0.0f, 0.0f),
        make_float4(0.0f, 0.0f, 0.0f, 0.0f),
        make_float4(0.0f, 0.0f, 0.0f, 0.0f),
        make_float4(0.0f, 0.0f, 0.0f, 0.0f),
    };

    size_t tileCount = M / blockDim.x;

    // (**)
    for (size_t tileIndex = 0; tileIndex < tileCount; tileIndex++)
    {
        // (***)
#pragma unroll
        for (int r = 0; r < 4; r++)
        {
            ATile(4 * tileRow + r, tileColumn) = A(globalRowBase + r, tileIndex * blockDim.x + tileColumn);
        }

        // (***)
        *reinterpret_cast<float4*>(&BTile(tileRow, 4 * tileColumn)) =
            *reinterpret_cast<const float4*>(&B(tileIndex * blockDim.x + tileRow, globalColumnBase));
        __syncthreads();

        for (size_t k = 0; k < blockDim.x; k++)
        {
            auto bV = *reinterpret_cast<float4*>(&BTile(k, 4 * tileColumn));

#pragma unroll
            for (int r = 0; r < 4; r++)
            {
                float aV =
                    ATile(tileRow * 4 + r, k);

                accumulatedValues[r].x += aV * bV.x;
                accumulatedValues[r].y += aV * bV.y;
                accumulatedValues[r].z += aV * bV.z;
                accumulatedValues[r].w += aV * bV.w;
            }
        }
        __syncthreads();
    }

    // (***)
#pragma unroll
    for (int r = 0; r < 4; r++) {
        float4 cV =
            *reinterpret_cast<const float4*>(
                &C(globalRowBase + r, globalColumnBase));

        float4 v = make_float4(
            alpha * accumulatedValues[r].x + beta * cV.x,
            alpha * accumulatedValues[r].y + beta * cV.y,
            alpha * accumulatedValues[r].z + beta * cV.z,
            alpha * accumulatedValues[r].w + beta * cV.w
        );

        *reinterpret_cast<float4*>(
            &C(globalRowBase + r, globalColumnBase)) = v;
    }
}


namespace blas
{

milliseconds
gemm(const execution::parallel_gpu::own &,
     size_t N, size_t M, size_t P,
     float * result,
     float alpha, const float * a, const float * b,
     float beta, const float * c)
{
    float * aDevice;
    float * bDevice;
    float * cDevice;
    cudaAssert(cudaMalloc(&aDevice, N * M * sizeof(float)));
    cudaAssert(cudaMalloc(&bDevice, M * P * sizeof(float)));
    cudaAssert(cudaMalloc(&cDevice, N * P * sizeof(float)));

    cudaAssert(cudaMemcpy(aDevice, a, N * M * sizeof(float), cudaMemcpyHostToDevice));
    cudaAssert(cudaMemcpy(bDevice, b, M * P * sizeof(float), cudaMemcpyHostToDevice));
    cudaAssert(cudaMemcpy(cDevice, c, N * P * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t startGpu;
    cudaEvent_t stopGpu;
    cudaAssert(cudaEventCreate(&startGpu));
    cudaAssert(cudaEventCreate(&stopGpu));

    const auto ceilDiv = [](size_t v, size_t d)
    {
        return (v + d - 1)/d;
    };

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    dim3 blockSize(prop.warpSize, prop.warpSize);
    dim3 gridSize(ceilDiv(N, blockSize.x * 4), ceilDiv(M, blockSize.y * 4), 1);
    size_t sharedMemorySize = (prop.warpSize * prop.warpSize * 4 * sizeof(float)) +
        (prop.warpSize * prop.warpSize * 4 * sizeof(float)); // two shared tiles - one for A, one for B both multiplied by 4
    size_t sharedMemMax = prop.sharedMemPerBlock;
    cudaAssert(cudaEventRecord(startGpu));
    gemm_kernel<<<gridSize, blockSize, sharedMemorySize>>>(
        N, M, P,
        alpha,
        aDevice, bDevice,
        beta, cDevice
    );
    cudaAssert(cudaEventRecord(stopGpu));

    cudaAssert(cudaMemcpy(result, cDevice, N * P * sizeof(float), cudaMemcpyDeviceToHost));

    float gpuMs;
    cudaAssert(cudaEventElapsedTime(&gpuMs, startGpu, stopGpu));

    cudaAssert(cudaDeviceSynchronize());

    cudaAssert(cudaEventDestroy(stopGpu));
    cudaAssert(cudaEventDestroy(startGpu));

    cudaAssert(cudaFree(aDevice));
    cudaAssert(cudaFree(bDevice));
    cudaAssert(cudaFree(cDevice));

    cudaAssert(cudaDeviceSynchronize());

    return milliseconds{gpuMs};
}

milliseconds
gemm(const execution::parallel_gpu::cublas &,
     size_t N, size_t M, size_t P,
     float * result,
     float alpha, const float * a, const float * b,
     float beta, const float * c)
{
    float * aDevice;
    float * bDevice;
    float * cDevice;
    cudaAssert(cudaMalloc(&aDevice, N * M * sizeof(float)));
    cudaAssert(cudaMalloc(&bDevice, M * P * sizeof(float)));
    cudaAssert(cudaMalloc(&cDevice, N * P * sizeof(float)));

    cudaAssert(cudaMemcpy(aDevice, a, N * M * sizeof(float), cudaMemcpyHostToDevice));
    cudaAssert(cudaMemcpy(bDevice, b, M * P * sizeof(float), cudaMemcpyHostToDevice));
    cudaAssert(cudaMemcpy(cDevice, c, N * P * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasAssert(cublasCreate(&handle));

    cudaEvent_t startGpu;
    cudaEvent_t stopGpu;
    cudaAssert(cudaEventCreate(&startGpu));
    cudaAssert(cudaEventCreate(&stopGpu));

    cudaAssert(cudaEventRecord(startGpu));
    cublasAssert(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        P, N, M,
        &alpha,
        bDevice, P,
        aDevice, M,
        &beta,
        cDevice, P
    ));
    cudaAssert(cudaEventRecord(stopGpu));

    cudaAssert(cudaMemcpy(result, cDevice, N * P * sizeof(float), cudaMemcpyDeviceToHost));

    float gpuMs;
    cudaAssert(cudaEventElapsedTime(&gpuMs, startGpu, stopGpu));

    cudaAssert(cudaDeviceSynchronize());

    cudaAssert(cudaEventDestroy(stopGpu));
    cudaAssert(cudaEventDestroy(startGpu));

    cudaAssert(cudaFree(aDevice));
    cudaAssert(cudaFree(bDevice));
    cudaAssert(cudaFree(cDevice));

    cublasAssert(cublasDestroy(handle));

    return milliseconds{gpuMs};
}

milliseconds
gemm(const execution::parallel_gpu::cublas_lt &,
     size_t N, size_t M, size_t P,
     float * result,
     float alpha, const float * a, const float * b,
     float beta, const float * c)
{
    cublasLtHandle_t ltHandle;
    cublasAssert(cublasLtCreate(&ltHandle));

    float *d_A, *d_B, *d_C;

    size_t sizeA = N * M * sizeof(float);
    size_t sizeB = M * P * sizeof(float);
    size_t sizeC = N * P * sizeof(float);

    cudaAssert(cudaMalloc(&d_A, sizeA));
    cudaAssert(cudaMalloc(&d_B, sizeB));
    cudaAssert(cudaMalloc(&d_C, sizeC));

    cudaAssert(cudaMemcpy(d_A, a, sizeA, cudaMemcpyHostToDevice));
    cudaAssert(cudaMemcpy(d_B, b, sizeB, cudaMemcpyHostToDevice));
    cudaAssert(cudaMemcpy(d_C, c, sizeC, cudaMemcpyHostToDevice));

    // Row-major fix: compute C^T = B^T * A^T
    int64_t rows = P;
    int64_t cols = N;
    int64_t kdim = M;

    cublasLtMatmulDesc_t operationDesc;
    cublasAssert(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F_PEDANTIC, CUDA_R_32F));

    cublasOperation_t opN = CUBLAS_OP_N;

    cublasAssert(cublasLtMatmulDescSetAttribute(
        operationDesc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &opN, sizeof(opN)));

    cublasAssert(cublasLtMatmulDescSetAttribute(
        operationDesc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &opN, sizeof(opN)));

    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;

    cublasAssert(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, rows, kdim, rows));
    cublasAssert(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F,kdim, cols, kdim));
    cublasAssert(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, rows, cols, rows));

    cublasLtMatmulPreference_t preference;
    cublasAssert(cublasLtMatmulPreferenceCreate(&preference));

    size_t workspaceSize = 1 << 22; // 4MB
    void* workspace;
    cudaAssert(cudaMalloc(&workspace, workspaceSize));

    cublasAssert(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize,
        sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;

    cublasAssert(cublasLtMatmulAlgoGetHeuristic(
        ltHandle,
        operationDesc,
        layoutA,
        layoutB,
        layoutC,
        layoutC,
        preference,
        1,
        &heuristicResult,
        &returnedResults));

    assert(returnedResults > 0);

    // ----------------------------
    // Timing (kernel only)
    // ----------------------------
    cudaEvent_t start, stop;
    cudaAssert(cudaEventCreate(&start));
    cudaAssert(cudaEventCreate(&stop));

    cudaAssert(cudaEventRecord(start));

    cublasAssert(cublasLtMatmul(
        ltHandle,
        operationDesc,
        &alpha,
        d_B, layoutA,
        d_A, layoutB,
        &beta,
        d_C, layoutC,
        d_C, layoutC,
        &heuristicResult.algo,
        workspace,
        workspaceSize,
        0));

    cudaAssert(cudaEventRecord(stop));
    cudaAssert(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    cudaAssert(cudaEventElapsedTime(&elapsedMs, start, stop));

    cudaAssert(cudaMemcpy(result, d_C, sizeC, cudaMemcpyDeviceToHost));

    cudaAssert(cudaEventDestroy(start));
    cudaAssert(cudaEventDestroy(stop));

    cudaAssert(cudaFree(workspace));
    cudaAssert(cudaFree(d_A));
    cudaAssert(cudaFree(d_B));
    cudaAssert(cudaFree(d_C));

    cublasAssert(cublasLtMatmulPreferenceDestroy(preference));
    cublasAssert(cublasLtMatrixLayoutDestroy(layoutA));
    cublasAssert(cublasLtMatrixLayoutDestroy(layoutB));
    cublasAssert(cublasLtMatrixLayoutDestroy(layoutC));
    cublasAssert(cublasLtMatmulDescDestroy(operationDesc));
    cublasAssert(cublasLtDestroy(ltHandle));

    return milliseconds(elapsedMs);
}

}
