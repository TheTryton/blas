#include <src_common.hpp>

#include <cassert>

namespace blas
{

void
cudaAssert(cudaError_t error)
{
    if (error != cudaSuccess) {
        assert(error == cudaSuccess);
    }
}

void
cublasAssert(cublasStatus_t error)
{
    if (error != CUBLAS_STATUS_SUCCESS) {
        assert(error == CUBLAS_STATUS_SUCCESS);
    }
}

}