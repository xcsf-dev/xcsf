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

#define BLOCK_SIZE 1024

#define CUDA_CALL(x) { cudaError_t cuda_error__ = (x); if(cuda_error__) { printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__)); } }

void cuda_fill(int N, double *X, double ALPHA, const cudaStream_t *stream);
double *cuda_make_array(const double *x, size_t n, const cudaStream_t *stream);
int cuda_get_device();
void cuda_info();
void cuda_set_device(int n);
void cuda_copy(int N, const double *X, double *Y, const cudaStream_t *stream);
int cuda_number_of_blocks(int array_size, int block_size);
