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
 
__global__ void gpu_gemm_nn(int M, int N, int K, double ALPHA,
        const double *A, int lda,
        const double *B, int ldb,
        double *C, int ldc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < M && j < N) {
        for(int k = 0; k < K; k++) {
            C[i*ldc+j] += ALPHA * A[i*lda+k] * B[k*ldb+j];
        }
    }
}

extern "C" void gemm_gpu(int TA, int TB, int M, int N, int K, double ALPHA,
        const double *A, int lda,
        const double *B, int ldb,
        double BETA,
        double *C, int ldc)
{
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            C[i*ldc+j] *= BETA;
        }
    }
    // allocate memory on the device
    double *d_a, *d_b, *d_c;
    CUDA_CALL( cudaMalloc((void **) &d_a, sizeof(double) * M * K) );
    CUDA_CALL( cudaMalloc((void **) &d_b, sizeof(double) * N * K) );
    CUDA_CALL( cudaMalloc((void **) &d_c, sizeof(double) * N * K) );

    // copy from host to device
    CUDA_CALL( cudaMemcpy(d_a, A, sizeof(double) * M * K, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_b, B, sizeof(double) * N * K, cudaMemcpyHostToDevice) );

    // run kernel on the GPU
    dim3 dimGrid(M,N);
    dim3 dimBlock(1,1);
    if(M > 65535) {
        dimBlock.x = sqrt(BLOCK_SIZE);
        dimBlock.y = dimBlock.x;
        dimGrid.x = (M % dimBlock.x == 0) ? M / dimBlock.x : (M / dimBlock.x) + 1;
    }

    if(!TA && !TB) {
        gpu_gemm_nn<<<dimGrid, dimBlock>>>(M,N,K,ALPHA,d_a,lda,d_b,ldb,d_c,ldc);
    }
    else {
        printf("TODO\n");
        exit(0);
    }

    // wait for GPU to finish
    CUDA_CALL( cudaDeviceSynchronize() );

    // copy result from device to host
    CUDA_CALL( cudaMemcpy(C, d_c, sizeof(double) * N*K, cudaMemcpyDeviceToHost) );

    // free memory
    CUDA_CALL( cudaFree(d_a) );
    CUDA_CALL( cudaFree(d_b) );
    CUDA_CALL( cudaFree(d_c) );
}
