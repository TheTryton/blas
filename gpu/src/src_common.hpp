#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>

namespace blas
{

void
cudaAssert(cudaError_t error);

void
cublasAssert(cublasStatus_t error);

}