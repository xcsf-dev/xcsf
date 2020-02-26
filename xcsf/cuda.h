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
 * @file cuda.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief General CUDA functions.
 */ 
 
#pragma once

#include <cuda_runtime.h>
#include "cublas_v2.h"

#define BLOCK_SIZE 1024

#define CUDA_CALL(x) { cudaError_t cuda_error__ = (x); if(cuda_error__) { printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__)); exit(0); } }

dim3 cuda_gridsize(size_t n);
cublasHandle_t blas_handle();

#ifdef __cplusplus
extern "C" {
#endif
void cuda_fill(int N, double *X, double ALPHA);
double *cuda_make_array(const double *x, size_t n);
int cuda_get_device();
void cuda_info();
void cuda_set_device(int n);
void cuda_copy(int N, const double *src, double *dest);
#ifdef __cplusplus
}
#endif
