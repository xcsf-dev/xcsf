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
 * @file blas.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief Basic linear algebra functions.
 */

#pragma once

void
blas_gemm(const int TA, const int TB, const int M, const int N, const int K,
          const double ALPHA, const double *A, const int lda, const double *B,
          const int ldb, const double BETA, double *C, const int ldc);

void
blas_axpy(const int N, const double ALPHA, const double *X, const int INCX,
          double *Y, const int INCY);

void
blas_mul(const int N, const double *X, const int INCX, double *Y,
         const int INCY);

void
blas_scal(const int N, const double ALPHA, double *X, const int INCX);

void
blas_fill(const int N, const double ALPHA, double *X, const int INCX);

double
blas_dot(const int N, const double *X, const int INCX, const double *Y,
         const int INCY);

double
blas_sum(const double *X, const int N);
