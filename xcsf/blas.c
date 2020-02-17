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
 * @file blas.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief Basic linear algebra functions.
 */ 
 
#include <stdio.h>

#ifdef GPU
#include "gemm_kernels.h"
#endif

static void gemm_nn(int M, int N, int K, double ALPHA,
        const double *A, int lda,
        const double *B, int ldb,
        double *C, int ldc)
{
    for(int i = 0; i < M; i++) {
        for(int k = 0; k < K; k++) {
            double A_PART = ALPHA * A[i*lda+k];
            for(int j = 0; j < N; j++) {
                C[i*ldc+j] += A_PART * B[k*ldb+j];
            }
        }
    }
}

static void gemm_nt(int M, int N, int K, double ALPHA,
        const double *A, int lda,
        const double *B, int ldb,
        double *C, int ldc)
{
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            double sum = 0;
            for(int k = 0; k < K; k++) {
                sum += ALPHA * A[i*lda+k] * B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

static void gemm_tn(int M, int N, int K, double ALPHA,
        const double *A, int lda,
        const double *B, int ldb,
        double *C, int ldc)
{
    for(int i = 0; i < M; i++) {
        for(int k = 0; k < K; k++) {
            double A_PART = ALPHA * A[k*lda+i];
            for(int j = 0; j < N; j++) {
                C[i*ldc+j] += A_PART * B[k*ldb+j];
            }
        }
    }
}

static void gemm_tt(int M, int N, int K, double ALPHA,
        const double *A, int lda,
        const double *B, int ldb,
        double *C, int ldc)
{
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            double sum = 0;
            for(int k = 0; k < K; k++) {
                sum += ALPHA * A[i+k*lda] * B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}

static void gemm_cpu(int TA, int TB, int M, int N, int K, double ALPHA,
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
    if(!TA && !TB) {
        gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    else if(TA && !TB) {
        gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    else if(!TA && TB) {
        gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    else {
        gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
}

static void axpy_cpu(int N, double ALPHA, const double *X, int INCX, double *Y, int INCY)
{
    for(int i = 0; i < N; i++) {
        Y[i*INCY] += ALPHA * X[i*INCX];
    }
}

static void scal_cpu(int N, double ALPHA, double *X, int INCX)
{
    for(int i = 0; i < N; i++) {
        X[i*INCX] *= ALPHA;
    }
}

static void fill_cpu(int N, double ALPHA, double *X, int INCX)
{
    for(int i = 0; i < N; i++) {
        X[i*INCX] = ALPHA;
    }
}

static double dot_cpu(int N, const double *X, int INCX, const double *Y, int INCY)
{
    double dot = 0;
    for(int i = 0; i < N; i++) {
        dot += X[i*INCX] * Y[i*INCY];
    }
    return dot;
}

void blas_gemm(int TA, int TB, int M, int N, int K, double ALPHA,
        const double *A, int lda,
        const double *B, int ldb,
        double BETA,
        double *C, int ldc)
{
#ifdef GPU
    gemm_gpu(TA,TB,M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
#else
    gemm_cpu(TA,TB,M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
#endif
}

void blas_axpy(int N, double ALPHA, const double *X, int INCX, double *Y, int INCY)
{
    axpy_cpu(N,ALPHA,X,INCX,Y,INCY);
}

void blas_scal(int N, double ALPHA, double *X, int INCX)
{
    scal_cpu(N,ALPHA,X,INCX);
}

void blas_fill(int N, double ALPHA, double *X, int INCX)
{
    fill_cpu(N,ALPHA,X,INCX);
}

double blas_dot(int N, const double *X, int INCX, const double *Y, int INCY)
{
    return dot_cpu(N,X,INCX,Y,INCY);
}
