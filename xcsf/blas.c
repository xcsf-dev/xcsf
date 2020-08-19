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

#include "blas.h"

static void
gemm_nn(const int M, const int N, const int K, const double ALPHA,
        const double *A, const int lda, const double *B, const int ldb,
        double *C, const int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            double A_PART = ALPHA * A[i * lda + k];
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

static void
gemm_nt(const int M, const int N, const int K, const double ALPHA,
        const double *A, const int lda, const double *B, const int ldb,
        double *C, const int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}

static void
gemm_tn(const int M, const int N, const int K, const double ALPHA,
        const double *A, const int lda, const double *B, const int ldb,
        double *C, const int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            double A_PART = ALPHA * A[k * lda + i];
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

static void
gemm_tt(const int M, const int N, const int K, const double ALPHA,
        const double *A, const int lda, const double *B, const int ldb,
        double *C, const int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}

/**
 * @brief Performs the matrix-matrix multiplication:
 * \f$ C = \alpha \mbox{op}(A) \mbox{op}(B) + \beta C \f$.
 * @param TA Operation op(A) that is non- or (conj.) transpose.
 * @param TB Operation op(B) that is non- or (conj.) transpose.
 * @param M Number of rows of matrix op(A) and C.
 * @param N Number of columns of matrix op(B) and C.
 * @param K Number of columns of op(A) and rows of op(B).
 * @param ALPHA Scalar used for multiplication.
 * @param A Array of dimension lda × K with lda >= max(1,M) if TA=0 and lda × M
 * with lda >= max(1,K) otherwise.
 * @param lda Leading dimension of a 2-D array used to store the matrix A.
 * @param B Array of dimension ldb × N with ldb >= max(1,K) if TB=0 and ldb × K
 * with ldb >= max(1,N) otherwise.
 * @param ldb Leading dimension of a 2-D array used to store the matrix B.
 * @param BETA Scalar used for multiplication.
 * @param C Array of dimension ldc × N with ldc >= max(1,M).
 * @param ldc Leading dimension of a 2-D array used to store the matrix C.
 */
void
blas_gemm(const int TA, const int TB, const int M, const int N, const int K,
          const double ALPHA, const double *A, const int lda, const double *B,
          const int ldb, const double BETA, double *C, const int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] *= BETA;
        }
    }
    if (!TA && !TB) {
        gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    } else if (TA && !TB) {
        gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    } else if (!TA && TB) {
        gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    } else {
        gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
}

/**
 * @brief Multiplies vector X by the scalar ALPHA and adds it to the vector Y.
 * @param N The number of elements in vectors X and Y.
 * @param ALPHA Scalar used for multiplication.
 * @param X Vector with N elements.
 * @param INCX Stride between consecutive elements of X.
 * @param Y Vector with N elements.
 * @param INCY Stride between consecutive elements of Y.
 */
void
blas_axpy(const int N, const double ALPHA, const double *X, const int INCX,
          double *Y, const int INCY)
{
    if (ALPHA != 1) {
        for (int i = 0; i < N; ++i) {
            Y[i * INCY] += ALPHA * X[i * INCX];
        }
    } else {
        for (int i = 0; i < N; ++i) {
            Y[i * INCY] += X[i * INCX];
        }
    }
}

/**
 * @brief Scales vector X by the scalar ALPHA and overwrites it with the result.
 * @param N The number of elements in vector X.
 * @param ALPHA Scalar used for multiplication.
 * @param X Vector with N elements.
 * @param INCX Stride between consecutive elements of X.
 */
void
blas_scal(const int N, const double ALPHA, double *X, const int INCX)
{
    if (ALPHA != 0) {
        for (int i = 0; i < N; ++i) {
            X[i * INCX] *= ALPHA;
        }
    } else {
        for (int i = 0; i < N; ++i) {
            X[i * INCX] = 0;
        }
    }
}

/**
 * @brief Fills the vector X with the value ALPHA.
 * @param N The number of elements in vector X.
 * @param ALPHA The value to fill the vector.
 * @param X Vector with N elements.
 * @param INCX Stride between consecutive elements of X.
 */
void
blas_fill(const int N, const double ALPHA, double *X, const int INCX)
{
    for (int i = 0; i < N; ++i) {
        X[i * INCX] = ALPHA;
    }
}

/**
 * @brief Computes the dot product of two vectors.
 * @param N The number of elements in vectors X and Y.
 * @param X Vector with N elements.
 * @param INCX Stride between consecutive elements of X.
 * @param Y Vector with N elements.
 * @param INCY Stride between consecutive elements of Y.
 * @return The resulting dot product.
 */
double
blas_dot(const int N, const double *X, const int INCX, const double *Y,
         const int INCY)
{
    double dot = 0;
    for (int i = 0; i < N; ++i) {
        dot += X[i * INCX] * Y[i * INCY];
    }
    return dot;
}

/**
 * @brief Multiplies vector X by the vector Y and stores the result in vector Y.
 * @param N The number of elements in vectors X and Y.
 * @param X Vector with N elements.
 * @param INCX Stride between consecutive elements of X.
 * @param Y Vector with N elements.
 * @param INCY Stride between consecutive elements of Y.
 */
void
blas_mul(const int N, const double *X, const int INCX, double *Y,
         const int INCY)
{
    for (int i = 0; i < N; ++i) {
        Y[i * INCY] *= X[i * INCX];
    }
}

/**
 * @brief Returns the sum of the vector X.
 * @param X Vector with N elements.
 * @param N The number of elements in vector X.
 * @return The resulting sum.
 */
double
blas_sum(const double *X, const int N)
{
    double sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += X[i];
    }
    return sum;
}
