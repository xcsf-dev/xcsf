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
#include "data_structures.h"
#include "random.h"
#include "cl.h"
#include "pred_rls.h"

#define RLS_SCALE_FACTOR 1000.0
#define RLS_LAMBDA 1.0

void matrix_matrix_multiply(XCSF *xcsf, double *srca, double *srcb, double *dest, int n);
void matrix_vector_multiply(XCSF *xcsf, double *srcm, double *srcv, double *dest, int n);
void init_matrix(XCSF *xcsf, double *matrix, int n);
 
typedef struct PRED_RLS {
	int weights_length;
	double **weights;
	double *matrix;
	double *pre;
} PRED_RLS;
 
void pred_rls_init(XCSF *xcsf, CL *c)
{
	PRED_RLS *pred = malloc(sizeof(PRED_RLS));
	c->pred = pred;

	if(xcsf->PRED_TYPE == 3) {
		// offset(1) + n linear + n quadratic + n*(n-1)/2 mixed terms
		pred->weights_length = 1 + 2 * xcsf->num_x_vars + 
			xcsf->num_x_vars * (xcsf->num_x_vars - 1) / 2;
	}
	else {
		pred->weights_length = xcsf->num_x_vars + 1;
	}

	pred->weights = malloc(sizeof(double*) * xcsf->num_y_vars);
	for(int var = 0; var < xcsf->num_y_vars; var++) {
		pred->weights[var] = malloc(sizeof(double)*pred->weights_length);
	}
	for(int var = 0; var < xcsf->num_y_vars; var++) {
		pred->weights[var][0] = xcsf->XCSF_X0;
		for(int i = 1; i < pred->weights_length; i++) {
			pred->weights[var][i] = 0.0;
		}
	}

	// initialise gain matrix
	pred->matrix = malloc(sizeof(double)*pred->weights_length*pred->weights_length);
	init_matrix(xcsf, pred->matrix, pred->weights_length);

	// initialise current prediction
	pred->pre = malloc(sizeof(double) * xcsf->num_y_vars);
}
 	
void init_matrix(XCSF *xcsf, double *matrix, int n)
{
	(void)xcsf;
	for(int row = 0; row < n; row++) {
		for(int col = 0; col < n; col++) {
			if(row != col) {
				matrix[row*n+col] = 0.0;
			}
			else {
				matrix[row*n+col] = RLS_SCALE_FACTOR;
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
	memcpy(to_pred->pre, from_pred->pre, sizeof(double) * xcsf->num_y_vars);
}
 
void pred_rls_free(XCSF *xcsf, CL *c)
{
	PRED_RLS *pred = c->pred;
	for(int var = 0; var < xcsf->num_y_vars; var++) {
		free(pred->weights[var]);
	}
	free(pred->weights);
	free(pred->matrix);
	free(pred->pre);
	free(pred);
}
     
void pred_rls_update(XCSF *xcsf, CL *c, double *y, double *x)
{
	PRED_RLS *pred = c->pred;
	int n = pred->weights_length;
	int n_sqrd = n*n;
	double tmp_input[n];
	double tmp_vec[n];
	double tmp_matrix1[n_sqrd];
	double tmp_matrix2[n_sqrd];

	tmp_input[0] = xcsf->XCSF_X0;
	int index = 1;
	// linear terms
	for(int i = 0; i < xcsf->num_x_vars; i++) {
		tmp_input[index++] = x[i];
	}

	if(xcsf->PRED_TYPE == 3) {
		// quadratic terms
		for(int i = 0; i < xcsf->num_x_vars; i++) {
			for(int j = i; j < xcsf->num_x_vars; j++) {
				tmp_input[index++] = x[i] * x[j];
			}
		}
	}

	// determine gain vector = matrix * tmp_input
	matrix_vector_multiply(xcsf, pred->matrix, tmp_input, tmp_vec, n);

	// divide gain vector by lambda + tmp_vec
	double divisor = RLS_LAMBDA;
	for(int i = 0; i < n; i++) {
		divisor += tmp_input[i] * tmp_vec[i];
	}
	for(int i = 0; i < n; i++) {
		tmp_vec[i] /= divisor;
	}

	// update weights using the error
	// pre has been updated for the current state during set_pred()
	for(int var = 0; var < xcsf->num_y_vars; var++) {
		double error = y[var] - pred->pre[var]; // pred_compute(pred, x);
		for(int i = 0; i < n; i++) {
			pred->weights[var][i] += error * tmp_vec[i];
		}
	}

	// update gain matrix
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			double tmp = tmp_vec[i] * tmp_input[j];
			if(i == j) {
				tmp_matrix1[i*n+j] = 1.0 - tmp;
			}
			else {
				tmp_matrix1[i*n+j] = -tmp;
			}
		}
	}
	matrix_matrix_multiply(xcsf, tmp_matrix1, pred->matrix, tmp_matrix2, n);

	// divide gain matrix entries by lambda
	for(int row = 0; row < n; row++) {
		for(int col = 0; col < n; col++) {
			pred->matrix[row*n+col] = tmp_matrix2[row*n+col] / RLS_LAMBDA;
		}
	}
}

double *pred_rls_compute(XCSF *xcsf, CL *c, double *x)
{
	PRED_RLS *pred = c->pred;
	for(int var = 0; var < xcsf->num_y_vars; var++) {
		// first coefficient is offset
		double pre = xcsf->XCSF_X0 * pred->weights[var][0];
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

		pred->pre[var] = pre;
	}
	return pred->pre;
} 
 
double pred_rls_pre(XCSF *xcsf, CL *c, int p)
{
	(void)xcsf;
	PRED_RLS *pred = c->pred;
	return pred->pre[p];
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

void matrix_matrix_multiply(XCSF *xcsf, double *srca, double *srcb, double *dest, int n)
{
	(void)xcsf;
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			dest[i*n+j] = srca[i*n] * srcb[j];
			for(int k = 1; k < n; k++) {
				dest[i*n+j] += srca[i*n+k] * srcb[k*n+j];
			}
		}
	}
}

void matrix_vector_multiply(XCSF *xcsf, double *srcm, double *srcv, double *dest, int n)
{
	(void)xcsf;
	for(int i = 0; i < n; i++) {
		dest[i] = srcm[i*n] * srcv[0];
		for(int j = 1; j < n; j++) {
			dest[i] += srcm[i*n+j] * srcv[j];
		}
	}
}
