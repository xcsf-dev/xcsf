/*
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file blas_kernels.cu
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief CUDA basic linear algebra functions.
 */ 
 
#include <iostream>
#include <stdio.h>
#include "cuda.h"

__global__ void kernel_sub(int N, double *A, double *B, double *C)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
        C[i] = A[i] - B[i];
    }
}

extern "C" void sub_gpu(int N, double *A, double *B, double *C)
{
    kernel_sub<<<cuda_gridsize(N), BLOCK_SIZE>>>(N, A, B, C);
}

extern "C" void scal_gpu(int N, double ALPHA, double *X, int INCX)
{
    cublasHandle_t handle = blas_handle();
    cublasDscal(handle, N, &ALPHA, X, INCX);
}

extern "C" void axpy_gpu(int N, double ALPHA, const double *X, int INCX, double *Y, int INCY)
{
    cublasHandle_t handle = blas_handle();
    cublasDaxpy(handle, N, &ALPHA, X, INCX, Y, INCY);
}

extern "C" void dot_gpu(int N, const double *X, int INCX, const double *Y, int INCY, double *res)
{
    cublasHandle_t handle = blas_handle();
    cublasDdot(handle, N, X, INCX, Y, INCY, res);
}

extern "C" void gemm_gpu(int TA, int TB, int M, int N, int K, double ALPHA,
        const double *A, int lda,
        const double *B, int ldb,
        double BETA,
        double *C, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cublasDgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B, ldb, A, lda, &BETA, C, ldc);
}
