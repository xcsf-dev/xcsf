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
 * @file pred_rls.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Recursive least mean squares prediction functions.
 */

#include "pred_rls.h"
#include "blas.h"
#include "utils.h"

void
pred_rls_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct PredRLS *pred = malloc(sizeof(struct PredRLS));
    c->pred = pred;
    // set the length of weights per predicted variable
    if (xcsf->PRED_TYPE == PRED_TYPE_RLS_QUADRATIC) {
        // offset(1) + n linear + n quadratic + n*(n-1)/2 mixed terms
        pred->n = 1 + 2 * xcsf->x_dim + xcsf->x_dim * (xcsf->x_dim - 1) / 2;
    } else {
        pred->n = xcsf->x_dim + 1;
    }
    // initialise weights
    pred->n_weights = pred->n * xcsf->y_dim;
    pred->weights = calloc(pred->n_weights, sizeof(double));
    blas_fill(xcsf->y_dim, xcsf->PRED_X0, pred->weights, pred->n);
    // initialise gain matrix
    const int n_sqrd = pred->n * pred->n;
    pred->matrix = calloc(n_sqrd, sizeof(double));
    for (int i = 0; i < pred->n; ++i) {
        pred->matrix[i * pred->n + i] = xcsf->PRED_RLS_SCALE_FACTOR;
    }
    // initialise temporary storage for weight updating
    pred->tmp_input = malloc(sizeof(double) * pred->n);
    pred->tmp_vec = calloc(pred->n, sizeof(double));
    pred->tmp_matrix1 = calloc(n_sqrd, sizeof(double));
    pred->tmp_matrix2 = calloc(n_sqrd, sizeof(double));
}

void
pred_rls_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    pred_rls_init(xcsf, dest);
    const struct PredRLS *dest_pred = dest->pred;
    const struct PredRLS *src_pred = src->pred;
    memcpy(dest_pred->weights, src_pred->weights,
           sizeof(double) * src_pred->n_weights);
}

void
pred_rls_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    struct PredRLS *pred = c->pred;
    free(pred->weights);
    free(pred->matrix);
    free(pred->tmp_input);
    free(pred->tmp_vec);
    free(pred->tmp_matrix1);
    free(pred->tmp_matrix2);
    free(pred);
}

void
pred_rls_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                const double *y)
{
    (void) x;
    const struct PredRLS *pred = c->pred;
    const int n = pred->n;
    // gain vector = matrix * tmp_input
    const double *A = pred->matrix;
    const double *B = pred->tmp_input;
    double *C = pred->tmp_vec;
    blas_gemm(0, 0, n, 1, n, 1, A, n, B, 1, 0, C, 1);
    // divide gain vector by lambda + tmp_vec
    double divisor = blas_dot(n, pred->tmp_input, 1, pred->tmp_vec, 1);
    divisor = 1 / (divisor + xcsf->PRED_RLS_LAMBDA);
    blas_scal(n, divisor, pred->tmp_vec, 1);
    // update weights using the error
    for (int i = 0; i < xcsf->y_dim; ++i) {
        const double error = y[i] - c->prediction[i];
        blas_axpy(n, error, pred->tmp_vec, 1, &pred->weights[i * n], 1);
    }
    // update gain matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const double tmp = pred->tmp_vec[i] * pred->tmp_input[j];
            if (i == j) {
                pred->tmp_matrix1[i * n + j] = 1 - tmp;
            } else {
                pred->tmp_matrix1[i * n + j] = -tmp;
            }
        }
    }
    // tmp_matrix2 = tmp_matrix1 * pred_matrix
    A = pred->tmp_matrix1;
    B = pred->matrix;
    C = pred->tmp_matrix2;
    blas_gemm(0, 0, n, n, n, 1, A, n, B, n, 0, C, n);
    // divide gain matrix entries by lambda
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            pred->matrix[i * n + j] =
                pred->tmp_matrix2[i * n + j] / xcsf->PRED_RLS_LAMBDA;
        }
    }
}

void
pred_rls_compute(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    const struct PredRLS *pred = c->pred;
    const int n = pred->n;
    pred_transform_input(xcsf, x, pred->tmp_input);
    for (int i = 0; i < xcsf->y_dim; ++i) {
        c->prediction[i] =
            blas_dot(n, &pred->weights[i * n], 1, pred->tmp_input, 1);
    }
}

void
pred_rls_print(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct PredRLS *pred = c->pred;
    printf("RLS weights: ");
    const int n = pred->n;
    for (int i = 0; i < xcsf->y_dim; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f, ", pred->weights[i * n + j]);
        }
        printf("\n");
    }
}

_Bool
pred_rls_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                   const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

_Bool
pred_rls_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    (void) c;
    return false;
}

double
pred_rls_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct PredRLS *pred = c->pred;
    return pred->n_weights;
}

size_t
pred_rls_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    const struct PredRLS *pred = c->pred;
    size_t s = 0;
    s += fwrite(&pred->n, sizeof(int), 1, fp);
    s += fwrite(&pred->n_weights, sizeof(int), 1, fp);
    s += fwrite(pred->weights, sizeof(double), pred->n_weights, fp);
    const int n_sqrd = pred->n * pred->n;
    s += fwrite(pred->matrix, sizeof(double), n_sqrd, fp);
    return s;
}

size_t
pred_rls_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    pred_rls_init(xcsf, c);
    struct PredRLS *pred = c->pred;
    size_t s = 0;
    s += fread(&pred->n, sizeof(int), 1, fp);
    s += fread(&pred->n_weights, sizeof(int), 1, fp);
    s += fread(pred->weights, sizeof(double), pred->n_weights, fp);
    const int n_sqrd = pred->n * pred->n;
    s += fread(pred->matrix, sizeof(double), n_sqrd, fp);
    return s;
}
