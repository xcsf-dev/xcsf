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
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "blas.h"
#include "cl.h"
#include "prediction.h"
#include "pred_rls.h"

static void init_matrix(const XCSF *xcsf, double *matrix, int n);
                    
/**
 * @brief Recursive least mean squares prediction data structure.
 */ 
typedef struct PRED_RLS {
    int weights_length; //!< Total number of weights
    double **weights; //!< Weights used to compute prediction
    double *matrix; //!< Gain matrix used to update weights
    double *tmp_input; //!< Temporary storage for updating weights
    double *tmp_vec; //!< Temporary storage for updating weights
    double *tmp_matrix1; //!< Temporary storage for updating gain matrix
    double *tmp_matrix2; //!< Temporary storage for updating gain matrix
} PRED_RLS;

void pred_rls_init(const XCSF *xcsf, CL *c)
{
    PRED_RLS *pred = malloc(sizeof(PRED_RLS));
    c->pred = pred;
    // set length of weights
    if(xcsf->PRED_TYPE == PRED_TYPE_RLS_QUADRATIC) {
        // offset(1) + n linear + n quadratic + n*(n-1)/2 mixed terms
        pred->weights_length = 1 + 2 * xcsf->x_dim + 
            xcsf->x_dim * (xcsf->x_dim - 1) / 2;
    }
    else {
        pred->weights_length = xcsf->x_dim + 1;
    }
    // initialise weights
    pred->weights = malloc(sizeof(double*) * xcsf->y_dim);
    for(int var = 0; var < xcsf->y_dim; var++) {
        pred->weights[var] = malloc(sizeof(double) * pred->weights_length);
    }
    for(int var = 0; var < xcsf->y_dim; var++) {
        pred->weights[var][0] = xcsf->PRED_X0;
        for(int i = 1; i < pred->weights_length; i++) {
            pred->weights[var][i] = 0;
        }
    }
    // initialise gain matrix
    int len_sqrd = pred->weights_length * pred->weights_length;
    pred->matrix = malloc(sizeof(double) * len_sqrd);
    init_matrix(xcsf, pred->matrix, pred->weights_length);
    // initialise temporary storage for weight updating
    pred->tmp_input = malloc(sizeof(double) * pred->weights_length);
    pred->tmp_vec = malloc(sizeof(double) * pred->weights_length);
    pred->tmp_matrix1 = malloc(sizeof(double) * len_sqrd);
    pred->tmp_matrix2 = malloc(sizeof(double) * len_sqrd);
}

static void init_matrix(const XCSF *xcsf, double *matrix, int n)
{
    for(int row = 0; row < n; row++) {
        for(int col = 0; col < n; col++) {
            if(row != col) {
                matrix[row * n + col] = 0;
            }
            else {
                matrix[row * n + col] = xcsf->PRED_RLS_SCALE_FACTOR;
            }
        }
    }
}

void pred_rls_copy(const XCSF *xcsf, CL *to, const CL *from)
{
    pred_rls_init(xcsf, to);
    const PRED_RLS *to_pred = to->pred;
    const PRED_RLS *from_pred = from->pred;
    for(int var = 0; var < xcsf->y_dim; var++) {
        memcpy(to_pred->weights[var], from_pred->weights[var], 
                sizeof(double)*from_pred->weights_length);
    }
}

void pred_rls_free(const XCSF *xcsf, const CL *c)
{
    PRED_RLS *pred = c->pred;
    for(int var = 0; var < xcsf->y_dim; var++) {
        free(pred->weights[var]);
    }
    free(pred->weights);
    free(pred->matrix);
    free(pred->tmp_input);
    free(pred->tmp_vec);
    free(pred->tmp_matrix1);
    free(pred->tmp_matrix2);
    free(pred);
}

void pred_rls_update(const XCSF *xcsf, const CL *c, const double *x, const double *y)
{
    const PRED_RLS *pred = c->pred;
    int n = pred->weights_length;
    pred->tmp_input[0] = xcsf->PRED_X0;
    int index = 1;
    // linear terms
    for(int i = 0; i < xcsf->x_dim; i++) {
        pred->tmp_input[index++] = x[i];
    }
    // quadratic terms
    if(xcsf->PRED_TYPE == PRED_TYPE_RLS_QUADRATIC) {
        for(int i = 0; i < xcsf->x_dim; i++) {
            for(int j = i; j < xcsf->x_dim; j++) {
                pred->tmp_input[index++] = x[i] * x[j];
            }
        }
    }
    // determine gain vector = matrix * tmp_input
    blas_gemm(0, 0, n, 1, n, 1, pred->matrix, n, pred->tmp_input, 1, 0, pred->tmp_vec, 1);
    // divide gain vector by lambda + tmp_vec
    double divisor = xcsf->PRED_RLS_LAMBDA;
    for(int i = 0; i < n; i++) {
        divisor += pred->tmp_input[i] * pred->tmp_vec[i];
    }
    for(int i = 0; i < n; i++) {
        pred->tmp_vec[i] /= divisor;
    }
    // update weights using the error
    // prediction must have been computed for the current state
    for(int var = 0; var < xcsf->y_dim; var++) {
        double error = y[var] - c->prediction[var];
        for(int i = 0; i < n; i++) {
            pred->weights[var][i] += error * pred->tmp_vec[i];
        }
    }
    // update gain matrix
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            double tmp = pred->tmp_vec[i] * pred->tmp_input[j];
            if(i == j) {
                pred->tmp_matrix1[i*n+j] = 1 - tmp;
            }
            else {
                pred->tmp_matrix1[i*n+j] = -tmp;
            }
        }
    }
    // tmp_matrix2 = tmp_matrix1 * pred_matrix
    blas_gemm(0, 0, n, n, n, 1, pred->tmp_matrix1, n, pred->matrix, n, 0, pred->tmp_matrix2, n);
    // divide gain matrix entries by lambda
    for(int row = 0; row < n; row++) {
        for(int col = 0; col < n; col++) {
            pred->matrix[row*n+col] = pred->tmp_matrix2[row*n+col] / xcsf->PRED_RLS_LAMBDA;
        }
    }
}

void pred_rls_compute(const XCSF *xcsf, const CL *c, const double *x)
{
    const PRED_RLS *pred = c->pred;
    for(int var = 0; var < xcsf->y_dim; var++) {
        // first coefficient is offset
        double pre = xcsf->PRED_X0 * pred->weights[var][0];
        int index = 1;
        // multiply linear coefficients with the prediction input
        for(int i = 0; i < xcsf->x_dim; i++) {
            pre += pred->weights[var][index++] * x[i];
        }
        if(xcsf->PRED_TYPE == PRED_TYPE_RLS_QUADRATIC) {
            // multiply quadratic coefficients with prediction input
            for(int i = 0; i < xcsf->x_dim; i++) {
                for(int j = i; j < xcsf->x_dim; j++) {
                    pre += pred->weights[var][index++] * x[i] * x[j];
                }
            }
        }
        c->prediction[var] = pre;
    }
} 

void pred_rls_print(const XCSF *xcsf, const CL *c)
{
    const PRED_RLS *pred = c->pred;
    printf("RLS weights: ");
    for(int var = 0; var < xcsf->y_dim; var++) {
        for(int i = 0; i < pred->weights_length; i++) {
            printf("%f, ", pred->weights[var][i]);
        }
        printf("\n");
    }
}

_Bool pred_rls_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool pred_rls_mutate(const XCSF *xcsf, const CL *c)
{
    (void)xcsf; (void)c;
    return false;
}

int pred_rls_size(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const PRED_RLS *pred = c->pred;
    return pred->weights_length;
}

size_t pred_rls_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    const PRED_RLS *pred = c->pred;
    size_t s = 0;
    s += fwrite(&pred->weights_length, sizeof(int), 1, fp);
    for(int var = 0; var < xcsf->y_dim; var++) {
        s += fwrite(pred->weights[var], sizeof(double), pred->weights_length, fp);
    }
    int len_sqrd = pred->weights_length * pred->weights_length;
    s += fwrite(pred->matrix, sizeof(double), len_sqrd, fp);
    return s;
}

size_t pred_rls_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    pred_rls_init(xcsf, c);
    PRED_RLS *pred = c->pred;
    size_t s = 0;
    s += fread(&pred->weights_length, sizeof(int), 1, fp);
    for(int var = 0; var < xcsf->y_dim; var++) {
        s += fread(pred->weights[var], sizeof(double), pred->weights_length, fp);
    }
    int len_sqrd = pred->weights_length * pred->weights_length;
    s += fread(pred->matrix, sizeof(double), len_sqrd, fp);
    return s;
}
