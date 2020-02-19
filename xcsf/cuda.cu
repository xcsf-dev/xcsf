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
 * @file cuda.cu
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief General CUDA functions.
 */ 

#include <stdio.h>
#include <stdint.h>
#include "blas_kernels.h"

extern "C" {
#include "cuda.h"
}

static void cuda_printDeviceInfo(cudaDeviceProp devProp);

int gpu_index = 0;

extern "C" void cuda_set_device(int n)
{
    gpu_index = n;
    CUDA_CALL( cudaSetDevice(n) );
}

extern "C" int cuda_get_device()
{
    int n = 0;
    CUDA_CALL( cudaGetDevice(&n) );
    return n;
}

extern "C" void cuda_free(double *x_gpu)
{
    CUDA_CALL( cudaFree(x_gpu) );
}

extern "C" void cuda_push_array(double *x_gpu, double *x, size_t n, const cudaStream_t *stream)
{
    CUDA_CALL( cudaMemcpyAsync(x_gpu, x, sizeof(double) * n, cudaMemcpyHostToDevice, *stream) );
}

extern "C" void cuda_pull_array(double *x_gpu, double *x, size_t n, const cudaStream_t *stream)
{
    CUDA_CALL( cudaMemcpyAsync(x, x_gpu, sizeof(double) * n, cudaMemcpyDeviceToHost, *stream) );
}

extern "C" double *cuda_make_array(double *x, size_t n, const cudaStream_t *stream)
{
    double *x_gpu;
    size_t size = sizeof(double) * n;
    CUDA_CALL( cudaMalloc((void **) &x_gpu, size) );
    if(x) {
        CUDA_CALL( cudaMemcpyAsync(x_gpu, x, size, cudaMemcpyHostToDevice, *stream) );
    }
    else {
        CUDA_CALL( cudaMemsetAsync(x_gpu, 0, size, *stream) );
    }
    return x_gpu;
}

extern "C" void cuda_create_stream(cudaStream_t *stream)
{
    CUDA_CALL( cudaStreamCreate(stream) );
}

extern "C" void cuda_destroy_stream(cudaStream_t *stream)
{
    CUDA_CALL( cudaStreamDestroy(*stream) );
}

extern "C" void cuda_memset(double *x_gpu, int value, size_t n, const cudaStream_t *stream)
{
    CUDA_CALL( cudaMemsetAsync(x_gpu, value, n, *stream) );
}

extern "C" void cuda_info()
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);
    for (int i = 0; i < devCount; i++) {
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        cuda_printDeviceInfo(devProp);
    }
}

static void cuda_printDeviceInfo(cudaDeviceProp devProp)
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
