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
 * @file gemm_kernels.cu
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief CUDA general matrix multiplication.
 */ 
 
#ifdef GPU

#include <iostream>

#define BLOCK_SIZE 8

#define CUDA_CALL(x) {
    cudaError_t cuda_error__ = (x);
    if(cuda_error__) {
        printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));
    }
}

/* grid and block should be configured as:
 * dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
 * dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
 */
__global__ void kernel_mm_multiply(const double *A, const double *B, double *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    if(row < n && col < n) {
        for(int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
    }
    C[row * n + col] = sum;
}

__global__ void kernel_mv_multiply(const double *A, const double *B, double *C, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    if(row < n) {
        for(int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i];
        }
    }
    C[row] = sum;
}

extern "C" void gpu_mm_multiply(const double *A, const double *B, double *C, int n)
{
    int size = n*n;
    // allocate memory on the device
    double *d_a, *d_b, *d_c;
    CUDA_CALL( cudaMalloc((void **) &d_a, sizeof(double) * size) );
    CUDA_CALL( cudaMalloc((void **) &d_b, sizeof(double) * size) );
    CUDA_CALL( cudaMalloc((void **) &d_c, sizeof(double) * size) );

    // copy from host to device
    CUDA_CALL( cudaMemcpy(d_a, A, sizeof(double) * size, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_b, B, sizeof(double) * size, cudaMemcpyHostToDevice) );

    // run kernel on the GPU
    unsigned int grid_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocksPerGrid(grid_cols, grid_rows);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    kernel_mm_multiply<<<blocksPerGrid,threadsPerBlock>>>(d_a, d_b, d_c, n);

    // wait for GPU to finish
    CUDA_CALL( cudaDeviceSynchronize() );

    // copy result from device to host
    CUDA_CALL( cudaMemcpy(C, d_c, sizeof(double) * size, cudaMemcpyDeviceToHost) );

    // free memory
    CUDA_CALL( cudaFree(d_a) );
    CUDA_CALL( cudaFree(d_b) );
    CUDA_CALL( cudaFree(d_c) );
}

extern "C" void gpu_mv_multiply(const double *A, const double *B, double *C, int n)
{
    int size = n*n;
    // allocate memory on the device
    double *d_a, *d_b, *d_c;
    CUDA_CALL( cudaMalloc((void **) &d_a, sizeof(double) * size) );
    CUDA_CALL( cudaMalloc((void **) &d_b, sizeof(double) * n) );
    CUDA_CALL( cudaMalloc((void **) &d_c, sizeof(double) * n) );

    // copy from host to device
    CUDA_CALL( cudaMemcpy(d_a, A, sizeof(double) * size, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_b, B, sizeof(double) * n, cudaMemcpyHostToDevice) );

    // run kernel on the GPU
    unsigned int grid_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocksPerGrid(grid_cols, grid_rows);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    kernel_mv_multiply<<<blocksPerGrid,threadsPerBlock>>>(d_a, d_b, d_c, n);

    // wait for GPU to finish
    CUDA_CALL( cudaDeviceSynchronize() );

    // copy result from device to host
    CUDA_CALL( cudaMemcpy(C, d_c, sizeof(double) * n, cudaMemcpyDeviceToHost) );

    // free memory
    CUDA_CALL( cudaFree(d_a) );
    CUDA_CALL( cudaFree(d_b) );
    CUDA_CALL( cudaFree(d_c) );
}

#endif
