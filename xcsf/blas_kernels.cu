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

__global__ void kernel_axpy(int N, double ALPHA,
        const double *X, int OFFX, int INCX,
        double *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
        Y[OFFY+i*INCY] += ALPHA * X[OFFX+i*INCX];
    }
}

__global__ void kernel_scal(int N, double ALPHA, double *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
        X[i*INCX] *= ALPHA;
    }
}

__global__ void kernel_gemm_nn(int M, int N, int K, double ALPHA,
        const double *A, int lda,
        const double *B, int ldb,
        double BETA,
        double *C, int ldc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < M && j < N) {
        C[i*ldc+j] *= BETA;
        for(int k = 0; k < K; k++) {
            C[i*ldc+j] += ALPHA * A[i*lda+k] * B[k*ldb+j];
        }
    }
}

__global__ void kernel_gemm_nt(int M, int N, int K, double ALPHA,
        const double *A, int lda,
        const double *B, int ldb,
        double BETA,
        double *C, int ldc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < M && j < N) {
        C[i*ldc+j] *= BETA;
        for(int k = 0; k < K; k++) {
            C[i*ldc+j] += ALPHA * A[i*lda+k] * B[j*ldb+k];
        }
    }
}

__global__ void kernel_gemm_tn(int M, int N, int K, double ALPHA,
        const double *A, int lda,
        const double *B, int ldb,
        double BETA,
        double *C, int ldc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < M && j < N) {
        C[i*ldc+j] *= BETA;
        for(int k = 0; k < K; k++) {
            C[i*ldc+j] += ALPHA * A[k*lda+i] * B[k*ldb+j];
        }
    }
}

__global__ void kernel_gemm_tt(int M, int N, int K, double ALPHA,
        const double *A, int lda,
        const double *B, int ldb,
        double BETA,
        double *C, int ldc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < M && j < N) {
        C[i*ldc+j] *= BETA;
        for(int k = 0; k < K; k++) {
            C[i*ldc+j] += ALPHA * A[i+k*lda] * B[k+j*ldb];
        }
    }
}

extern "C" void gemm_gpu(int TA, int TB, int M, int N, int K, double ALPHA,
        const double *A, int lda,
        const double *B, int ldb,
        double BETA,
        double *C, int ldc,
        const cudaStream_t *stream)
{
    dim3 dimGrid(M,K);
    dim3 dimBlock(1,1);
    if(M > 65535) {
        dimBlock.x = sqrt(BLOCK_SIZE);
        dimBlock.y = dimBlock.x;
        dimGrid.x = (M % dimBlock.x == 0) ? M / dimBlock.x : (M / dimBlock.x) + 1;
    }
    if(!TA && !TB) {
        kernel_gemm_nn<<<dimGrid, dimBlock, 0, *stream>>>(M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
    }
    else if(TA && !TB) {
        kernel_gemm_tn<<<dimGrid, dimBlock, 0, *stream>>>(M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
    }
    else if(!TA && TB) {
        kernel_gemm_nt<<<dimGrid, dimBlock, 0, *stream>>>(M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
    }
    else {
        kernel_gemm_tt<<<dimGrid, dimBlock, 0, *stream>>>(M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
    }
}
