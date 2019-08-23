/*
 * Copyright (C) 2015--2019 Richard Preen <rpreen@gmail.com>
 *
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
 *
 **************
 * Description: 
 **************
 * The recursive least square classifier computed prediction module.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "prediction.h"
#include "pred_rls.h"

void matrix_matrix_multiply(double *srca, double *srcb, double *dest, int n);
void matrix_vector_multiply(double *srcm, double *srcv, double *dest, int n);
void init_matrix(XCSF *xcsf, double *matrix, int n);

typedef struct PRED_RLS {
    int weights_length;
    double **weights;
    double *matrix;
    // temporary storage for updating weights
    double *tmp_input;
    double *tmp_vec;
    double *tmp_matrix1;
    double *tmp_matrix2;
} PRED_RLS;

void pred_rls_init(XCSF *xcsf, CL *c)
{
    PRED_RLS *pred = malloc(sizeof(PRED_RLS));
    c->pred = pred;
    // set length of weights
    if(xcsf->PRED_TYPE == 3) {
        // offset(1) + n linear + n quadratic + n*(n-1)/2 mixed terms
        pred->weights_length = 1 + 2 * xcsf->num_x_vars + 
            xcsf->num_x_vars * (xcsf->num_x_vars - 1) / 2;
    }
    else {
        pred->weights_length = xcsf->num_x_vars + 1;
    }
    // initialise weights
    pred->weights = malloc(sizeof(double*) * xcsf->num_y_vars);
    for(int var = 0; var < xcsf->num_y_vars; var++) {
        pred->weights[var] = malloc(sizeof(double)*pred->weights_length);
    }
    for(int var = 0; var < xcsf->num_y_vars; var++) {
        pred->weights[var][0] = xcsf->X0;
        for(int i = 1; i < pred->weights_length; i++) {
            pred->weights[var][i] = 0.0;
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

void init_matrix(XCSF *xcsf, double *matrix, int n)
{
    for(int row = 0; row < n; row++) {
        for(int col = 0; col < n; col++) {
            if(row != col) {
                matrix[row*n+col] = 0.0;
            }
            else {
                matrix[row*n+col] = xcsf->RLS_SCALE_FACTOR;
            }
        }
    }
}

void pred_rls_copy(XCSF *xcsf, CL *to, CL *from)
{
    PRED_RLS *to_pred = to->pred;
    PRED_RLS *from_pred = from->pred;
    for(int var = 0; var < xcsf->num_y_vars; var++) {
        memcpy(to_pred->weights[var], from_pred->weights[var], 
                sizeof(double)*from_pred->weights_length);
    }
}

void pred_rls_free(XCSF *xcsf, CL *c)
{
    PRED_RLS *pred = c->pred;
    for(int var = 0; var < xcsf->num_y_vars; var++) {
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

void pred_rls_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    PRED_RLS *pred = c->pred;
    int n = pred->weights_length;
    pred->tmp_input[0] = xcsf->X0;
    int index = 1;
    // linear terms
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        pred->tmp_input[index++] = x[i];
    }
    // quadratic terms
    if(xcsf->PRED_TYPE == 3) {
        for(int i = 0; i < xcsf->num_x_vars; i++) {
            for(int j = i; j < xcsf->num_x_vars; j++) {
                pred->tmp_input[index++] = x[i] * x[j];
            }
        }
    }
    // determine gain vector = matrix * tmp_input
    matrix_vector_multiply(pred->matrix, pred->tmp_input, pred->tmp_vec, n);
    // divide gain vector by lambda + tmp_vec
    double divisor = xcsf->RLS_LAMBDA;
    for(int i = 0; i < n; i++) {
        divisor += pred->tmp_input[i] * pred->tmp_vec[i];
    }
    for(int i = 0; i < n; i++) {
        pred->tmp_vec[i] /= divisor;
    }
    // update weights using the error
    // prediction has been updated for the current state during set_pred()
    for(int var = 0; var < xcsf->num_y_vars; var++) {
        double error = y[var] - c->prediction[var]; // pred_compute();
        for(int i = 0; i < n; i++) {
            pred->weights[var][i] += error * pred->tmp_vec[i];
        }
    }
    // update gain matrix
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            double tmp = pred->tmp_vec[i] * pred->tmp_input[j];
            if(i == j) {
                pred->tmp_matrix1[i*n+j] = 1.0 - tmp;
            }
            else {
                pred->tmp_matrix1[i*n+j] = -tmp;
            }
        }
    }
    matrix_matrix_multiply(pred->tmp_matrix1, pred->matrix, pred->tmp_matrix2, n);

    // divide gain matrix entries by lambda
    for(int row = 0; row < n; row++) {
        for(int col = 0; col < n; col++) {
            pred->matrix[row*n+col] = pred->tmp_matrix2[row*n+col] / xcsf->RLS_LAMBDA;
        }
    }
}

double *pred_rls_compute(XCSF *xcsf, CL *c, double *x)
{
    PRED_RLS *pred = c->pred;
    for(int var = 0; var < xcsf->num_y_vars; var++) {
        // first coefficient is offset
        double pre = xcsf->X0 * pred->weights[var][0];
        int index = 1;
        // multiply linear coefficients with the prediction input
        for(int i = 0; i < xcsf->num_x_vars; i++) {
            pre += pred->weights[var][index++] * x[i];
        }

        if(xcsf->PRED_TYPE == 3) {
            // multiply quadratic coefficients with prediction input
            for(int i = 0; i < xcsf->num_x_vars; i++) {
                for(int j = i; j < xcsf->num_x_vars; j++) {
                    pre += pred->weights[var][index++] * x[i] * x[j];
                }
            }
        }

        c->prediction[var] = pre;
    }
    return c->prediction;
} 

void pred_rls_print(XCSF *xcsf, CL *c)
{
    PRED_RLS *pred = c->pred;
    printf("RLS weights: ");
    for(int var = 0; var < xcsf->num_y_vars; var++) {
        for(int i = 0; i < pred->weights_length; i++) {
            printf("%f, ", pred->weights[var][i]);
        }
        printf("\n");
    }
    //	printf("RLS matrix: ");
    //	int n = pred->weights_length;
    //	for(int i = 0; i < n; i++) {
    //		for(int j = 0; j < n; j++) {
    //			printf("%f, ", pred->matrix[i*n+j]);
    //		}
    //	}
    //	printf("\n");
}

void matrix_matrix_multiply( double *srca, double *srcb, double *dest, int n)
{
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            dest[i*n+j] = srca[i*n] * srcb[j];
            for(int k = 1; k < n; k++) {
                dest[i*n+j] += srca[i*n+k] * srcb[k*n+j];
            }
        }
    }
}

void matrix_vector_multiply(double *srcm, double *srcv, double *dest, int n)
{
    for(int i = 0; i < n; i++) {
        dest[i] = srcm[i*n] * srcv[0];
        for(int j = 1; j < n; j++) {
            dest[i] += srcm[i*n+j] * srcv[j];
        }
    }
}

_Bool pred_rls_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool pred_rls_mutate(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
    return false;
}
