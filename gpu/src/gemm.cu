#include <gpu/gemm.hpp>

#include <src_common.hpp>

namespace blas
{

milliseconds
gemm(const std::execution::parallel_gpu &,
     size_t N, size_t M, size_t P,
     float * result,
     float alpha, const float * a, const float * b,
     float beta, const float * c)
{
    cudaStream_t stream;
    cudaAssert(cudaStreamCreate(&stream));

    cublasLtHandle_t handle;
    cublasAssert(cublasLtCreate(&handle));

    cublasLtMatmulDesc_t operationDescription;
    cublasOperation_t trans = CUBLAS_OP_N;
    cublasAssert(cublasLtMatmulDescCreate(&operationDescription, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasAssert(cublasLtMatmulDescSetAttribute(operationDescription, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    cublasAssert(cublasLtMatmulDescSetAttribute(operationDescription, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));
    cublasAssert(cublasLtMatmulDescSetAttribute(operationDescription, CUBLASLT_MATMUL_DESC_TRANSC, &trans, sizeof(trans)));

    cublasLtMatrixLayout_t matrixADescription;
    cublasLtMatrixLayout_t matrixBDescription;
    cublasLtMatrixLayout_t matrixCDescription;
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    cublasAssert(cublasLtMatrixLayoutCreate(&matrixADescription, CUDA_R_32F, N, M, M));
    cublasAssert(cublasLtMatrixLayoutSetAttribute(matrixADescription, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    cublasAssert(cublasLtMatrixLayoutCreate(&matrixBDescription, CUDA_R_32F, M, P, P));
    cublasAssert(cublasLtMatrixLayoutSetAttribute(matrixBDescription, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    cublasAssert(cublasLtMatrixLayoutCreate(&matrixCDescription, CUDA_R_32F, N, P, P));
    cublasAssert(cublasLtMatrixLayoutSetAttribute(matrixCDescription, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    cublasLtMatmulPreference_t preference;
    size_t workspaceSize = 1024 * 1024 * sizeof(float );
    cublasAssert(cublasLtMatmulPreferenceCreate(&preference));
    cublasAssert(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    int returnedResults{};
    cublasLtMatmulHeuristicResult_t heuristicResult;
    cublasAssert(cublasLtMatmulAlgoGetHeuristic(
        handle, operationDescription,
        matrixADescription, matrixBDescription, matrixCDescription, matrixCDescription,
        preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        cublasAssert(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    float * aDevice;
    float * bDevice;
    float * cDevice;
    cudaAssert(cudaMalloc(&aDevice, N * M * sizeof(float)));
    cudaAssert(cudaMalloc(&bDevice, M * P * sizeof(float)));
    cudaAssert(cudaMalloc(&cDevice, N * P * sizeof(float)));

    float * workspace;
    cudaAssert(cudaMalloc(&workspace, workspaceSize));

    cudaAssert(cudaMemcpyAsync(aDevice, a, N * M * sizeof(float), cudaMemcpyHostToDevice, stream));
    cudaAssert(cudaMemcpyAsync(bDevice, b, M * P * sizeof(float), cudaMemcpyHostToDevice, stream));
    cudaAssert(cudaMemcpyAsync(cDevice, c, N * P * sizeof(float), cudaMemcpyHostToDevice, stream));

    cudaEvent_t startGpu;
    cudaEvent_t stopGpu;
    cudaAssert(cudaEventCreate(&startGpu));
    cudaAssert(cudaEventCreate(&stopGpu));

    cudaAssert(cudaEventRecord(startGpu, stream));
    cublasAssert(cublasLtMatmul(handle,
                                operationDescription,
                                &alpha,
                                aDevice,
                                matrixADescription,
                                bDevice,
                                matrixBDescription,
                                &beta,
                                cDevice,
                                matrixCDescription,
                                cDevice,
                                matrixCDescription,
                                &heuristicResult.algo,
                                workspace,
                                workspaceSize,
                                stream));
    cudaAssert(cudaEventRecord(stopGpu, stream));

    cudaAssert(cudaMemcpyAsync(result, cDevice, N * P * sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaAssert(cudaStreamSynchronize(stream));

    float gpuMs;
    cudaAssert(cudaEventElapsedTime(&gpuMs, startGpu, stopGpu));

    cudaAssert(cudaDeviceSynchronize());

    cudaAssert(cudaEventDestroy(stopGpu));
    cudaAssert(cudaEventDestroy(startGpu));

    cudaAssert(cudaFree(workspace));

    cudaAssert(cudaFree(aDevice));
    cudaAssert(cudaFree(bDevice));
    cudaAssert(cudaFree(cDevice));


    cublasAssert(cublasLtMatmulPreferenceDestroy(preference));
    cublasAssert(cublasLtMatrixLayoutDestroy(matrixCDescription));
    cublasAssert(cublasLtMatrixLayoutDestroy(matrixBDescription));
    cublasAssert(cublasLtMatrixLayoutDestroy(matrixADescription));
    cublasAssert(cublasLtMatmulDescDestroy(operationDescription));
    cublasAssert(cublasLtDestroy(handle));

    cudaAssert(cudaStreamDestroy(stream));

    return milliseconds{gpuMs};

    /*
    cublasAssert(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED));
    cublasAssert(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    cublasAssert(cublasSetStream(handle, stream));
    cublasAssert(order(handle, stream));

    float * aDevice;
    float * bDevice;
    float * cDevice;
    cudaAssert(cudaMalloc(&aDevice, N * M * sizeof(float)));
    cudaAssert(cudaMalloc(&bDevice, M * P * sizeof(float)));
    cudaAssert(cudaMalloc(&cDevice, N * P * sizeof(float)));

    float * aHost;
    float * bHost;
    float * cHost;
    cudaAssert(cudaMallocHost(&aHost, N * M * sizeof(float)));
    cudaAssert(cudaMallocHost(&bHost, M * P * sizeof(float)));
    cudaAssert(cudaMallocHost(&cHost, N * P * sizeof(float)));



    cublasAssert(cublasSetMatrixAsync(N, M, sizeof(float), a, M, aDevice, M, stream));
    cublasAssert(cublasSetMatrixAsync(M, P, sizeof(float), b, P, bDevice, P, stream));
    cublasAssert(cublasSetMatrixAsync(N, P, sizeof(float), c, P, cDevice, P, stream));

    cudaEvent_t startGpu;
    cudaEvent_t stopGpu;
    cudaAssert(cudaEventCreate(&startGpu));
    cudaAssert(cudaEventCreate(&stopGpu));

    cublasAssert(
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, P, M, &alpha, aDevice, CUDA_R_32F, M, bDevice,
                     CUDA_R_32F, P,
                     &beta, cDevice, CUDA_R_32F, P, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        ));
    cudaAssert(cudaEventRecord(startGpu, stream));
    cublasAssert(
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, P, M, &alpha, aDevice, CUDA_R_32F, M, bDevice,
                     CUDA_R_32F, P,
                     &beta, cDevice, CUDA_R_32F, P, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        ));
    cudaAssert(cudaEventRecord(stopGpu, stream));

    cublasAssert(cublasGetMatrixAsync(N, P, sizeof(float), cDevice, P, result, P, stream));

    cudaAssert(cudaStreamSynchronize(stream));

    float gpuMs;
    cudaAssert(cudaEventElapsedTime(&gpuMs, startGpu, stopGpu));

    cudaAssert(cudaDeviceSynchronize());

    cudaAssert(cudaEventDestroy(stopGpu));
    cudaAssert(cudaEventDestroy(startGpu));

    cudaAssert(cudaFree(aDevice));
    cudaAssert(cudaFree(bDevice));
    cudaAssert(cudaFree(cDevice));

    cublasAssert(cublasDestroy(handle));

    cudaAssert(cudaStreamDestroy(stream));

     return milliseconds{gpuMs};
*/
}

}
