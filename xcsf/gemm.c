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
 * @file gemm.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief General matrix multiplication.
 */ 
 
#include "gemm_kernels.h"

static void cpu_mm_multiply(const double *A, const double *B, double *C, int n);
static void cpu_mv_multiply(const double *A, const double *B, double *C, int n);

void matrix_matrix_multiply(const double *A, const double *B, double *C, int n)
{
#ifdef GPU
    gpu_mm_multiply(A, B, C, n);
#else
    cpu_mm_multiply(A, B, C, n);
#endif
}

void matrix_vector_multiply(const double *A, const double *B, double *C, int n)
{
#ifdef GPU
    gpu_mv_multiply(A, B, C, n);
#else
    cpu_mv_multiply(A, B, C, n);
#endif
}

static void cpu_mm_multiply(const double *A, const double *B, double *C, int n)
{
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            C[i*n+j] = A[i*n] * B[j];
            for(int k = 1; k < n; k++) {
                C[i*n+j] += A[i*n+k] * B[k*n+j];
            }
        }
    }
}

static void cpu_mv_multiply(const double *A, const double *B, double *C, int n)
{
    for(int i = 0; i < n; i++) {
        C[i] = A[i*n] * B[0];
        for(int j = 1; j < n; j++) {
            C[i] += A[i*n+j] * B[j];
        }
    }
}
