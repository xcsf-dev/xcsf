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
extern "C" {
#include "cuda.h"
}

__device__ double atomic_Add(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void kernel_axpy(int N, double ALPHA, const double *X, int INCX, double *Y, int INCY)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
        Y[i*INCY] += ALPHA * X[i*INCX];
    }
}

__global__ void kernel_scal(int N, double ALPHA, double *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
        X[i*INCX] *= ALPHA;
    }
}

__global__ void kernel_sub(int N, double *A, double *B, double *C)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
        C[i] = A[i] - B[i];
    }
}

__global__ void kernel_dot(const double *A, const double *B, double *C, int N)
{
    __shared__ double cache;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    double temp = 0;
    cache = 0;
    __syncthreads();
    while (tid < N) {
        temp += A[tid] * B[tid];
        tid += blockDim.x * gridDim.x;
    }
    atomic_Add(&cache, temp);
    __syncthreads();
    if (cacheIndex == 0) {
        C[blockIdx.x] = cache;
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

extern "C" void sub_gpu(int N, double *A, double *B, double *C, const cudaStream_t *stream)
{
    const int num_blocks = cuda_number_of_blocks(N, BLOCK_SIZE);
    kernel_sub<<<num_blocks, BLOCK_SIZE, 0, *stream>>>(N, A, B, C);
}

extern "C" void scal_gpu(int N, double ALPHA, double *X, int INCX, const cudaStream_t *stream)
{
    const int num_blocks = cuda_number_of_blocks(N, BLOCK_SIZE);
    kernel_scal<<<num_blocks, BLOCK_SIZE, 0, *stream>>>(N, ALPHA, X, INCX);
}

extern "C" void axpy_gpu(int N, double ALPHA, const double *X, int INCX, double *Y, int INCY,
        const cudaStream_t *stream)
{
    const int num_blocks = cuda_number_of_blocks(N, BLOCK_SIZE);
    kernel_axpy<<<num_blocks, BLOCK_SIZE, 0, *stream>>>(N, ALPHA, X, INCX, Y, INCY);
}

extern "C" void dot_gpu(int N, const double *A, const double *B, double *C,
        const cudaStream_t *stream)
{
    const int num_blocks = cuda_number_of_blocks(N, BLOCK_SIZE);
    kernel_dot<<<num_blocks, BLOCK_SIZE, 0, *stream>>>(A, B, C, N);
}

extern "C" void gemm_gpu(int TA, int TB, int M, int N, int K, double ALPHA,
        const double *A, int lda,
        const double *B, int ldb,
        double BETA,
        double *C, int ldc,
        const cudaStream_t *stream)
{
    dim3 dimGrid(M,N);
    dim3 dimBlock(1,1);
    if(!TA && !TB) {
        kernel_gemm_nn<<<dimGrid,dimBlock,0,*stream>>>(M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
    }
    else if(TA && !TB) {
        kernel_gemm_tn<<<dimGrid,dimBlock,0,*stream>>>(M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
    }
    else if(!TA && TB) {
        kernel_gemm_nt<<<dimGrid,dimBlock,0,*stream>>>(M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
    }
    else {
        kernel_gemm_tt<<<dimGrid,dimBlock,0,*stream>>>(M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
    }
}
