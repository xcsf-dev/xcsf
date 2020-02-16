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

#define BLOCK_SIZE 1024

#define CUDA_CALL(x) {
    cudaError_t cuda_error__ = (x);
    if(cuda_error__) {
        printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));
    }
}

static void printDeviceInfo(cudaDeviceProp devProp);

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
    dim3 dimGrid(n,n);
    dim3 dimBlock(1,1);
    if(n > 65535) {
        dimBlock.x = sqrt(BLOCK_SIZE);
        dimBlock.y = dimBlock.x;
        dimGrid.x = (n % dimBlock.x == 0) ? n / dimBlock.x : (n / dimBlock.x) + 1;
    }
    kernel_mm_multiply<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);

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
    dim3 dimGrid(n,n);
    dim3 dimBlock(1,1);
    if(n > 65535) {
        dimBlock.x = sqrt(BLOCK_SIZE);
        dimBlock.y = dimBlock.x;
        dimGrid.x = (n % dimBlock.x == 0) ? n / dimBlock.x : (n / dimBlock.x) + 1;
    }
    kernel_mv_multiply<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);

    // wait for GPU to finish
    CUDA_CALL( cudaDeviceSynchronize() );

    // copy result from device to host
    CUDA_CALL( cudaMemcpy(C, d_c, sizeof(double) * n, cudaMemcpyDeviceToHost) );

    // free memory
    CUDA_CALL( cudaFree(d_a) );
    CUDA_CALL( cudaFree(d_b) );
    CUDA_CALL( cudaFree(d_c) );
}

extern "C" void gpu_info()
{
   int devCount;
   cudaGetDeviceCount(&devCount);
   printf("CUDA Device Query...\n");
   printf("There are %d CUDA devices.\n", devCount);
   for (int i = 0; i < devCount; i++) {
       printf("\nCUDA Device #%d\n", i);
       cudaDeviceProp devProp;
       cudaGetDeviceProperties(&devProp, i);
       printDeviceInfo(devProp);
   }
}

static void printDeviceInfo(cudaDeviceProp devProp)
{
    printf("Revision number:               %d.%d\n", devProp.major, devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu MB\n",  devProp.totalGlobalMem / (1024 * 1024));
    printf("Total shared memory per block: %lu kB\n",  devProp.sharedMemPerBlock / 1024);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu MB\n",  devProp.memPitch / (1024 * 1024));
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    printf("Maximum dimensions of block:   %d %d %d\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    printf("Maximum dimensions of grid:    %d %d %d\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
    printf("Clock rate:                    %d MHz\n",  devProp.clockRate / 1000);
    printf("Total constant memory:         %lu kB\n",  devProp.totalConstMem / 1024);
    printf("Texture alignment:             %lu B\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    printf("\n");
}

#endif
