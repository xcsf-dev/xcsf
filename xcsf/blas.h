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
blas_gemm(int TA, int TB, int M, int N, int K, double ALPHA,
          const double *A, int lda,
          const double *B, int ldb,
          double BETA,
          double *C, int ldc);

void
blas_axpy(int N, double ALPHA, const double *X, int INCX, double *Y, int INCY);

void
blas_mul(int N, const double *X, int INCX, double *Y, int INCY);

void
blas_scal(int N, double ALPHA, double *X, int INCX);

void
blas_fill(int N, double ALPHA, double *X, int INCX);

double
blas_dot(int N, const double *X, int INCX, const double *Y, int INCY);

double
blas_sum(const double *X, int N);
