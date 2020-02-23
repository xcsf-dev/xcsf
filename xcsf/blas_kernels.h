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
 * @file blas_kernels.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief CUDA basic linear algebra functions.
 */ 
 
#pragma once

void gemm_gpu(int TA, int TB, int M, int N, int K, double ALPHA,
        const double *A, int lda,
        const double *B, int ldb,
        double BETA,
        double *C, int ldc,
        const cudaStream_t *stream);

void scal_gpu(int N, double ALPHA, double *X, int INCX, const cudaStream_t *stream);
void sub_gpu(int N, double *A, double *B, double *C, const cudaStream_t *stream);
void dot_gpu(int N, const double *X, int INCX, const double *Y, int INCY, double *res,
        const cudaStream_t *stream);
void axpy_gpu(int N, double ALPHA, const double *X, int INCX, double *Y, int INCY,
        const cudaStream_t *stream);
