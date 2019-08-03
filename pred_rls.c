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

#if PRE == 2 || PRE == 3

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "cl.h"

#define RLS_SCALE_FACTOR 1000.0
#define RLS_LAMBDA 1.0

void matrix_matrix_multiply(double *srca, double *srcb, double *dest, int n);
void matrix_vector_multiply(double *srcm, double *srcv, double *dest, int n);
void init_matrix(double *matrix, int n);

void pred_init(CL *c)
{
	PRED *pred = &c->pred;
#if PRE == 3
	// offset(1) + n linear + n quadratic + n*(n-1)/2 mixed terms
	pred->weights_length = 1+2*num_x_vars+num_x_vars*(num_x_vars-1)/2;
#else
	pred->weights_length = num_x_vars+1;
#endif

	pred->weights = malloc(sizeof(double*)*num_y_vars);
	for(int var = 0; var < num_y_vars; var++) {
		pred->weights[var] = malloc(sizeof(double)*pred->weights_length);
	}
	for(int var = 0; var < num_y_vars; var++) {
		pred->weights[var][0] = XCSF_X0;
		for(int i = 1; i < pred->weights_length; i++) {
			pred->weights[var][i] = 0.0;
		}
	}

	// initialise gain matrix
	pred->matrix = malloc(sizeof(double)*pred->weights_length*pred->weights_length);
	init_matrix(pred->matrix, pred->weights_length);

	// initialise current prediction
	pred->pre = malloc(sizeof(double)*num_y_vars);
}
 	
void init_matrix(double *matrix, int n)
{
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
 
void pred_copy(CL *to, CL *from)
{
	for(int var = 0; var < num_y_vars; var++) {
		memcpy(to->pred.weights[var], from->pred.weights[var], 
				sizeof(double)*from->pred.weights_length);
	}
	memcpy(to->pred.pre, from->pred.pre, sizeof(double)*num_y_vars);
}
 
void pred_free(CL *c)
{
	for(int var = 0; var < num_y_vars; var++) {
		free(c->pred.weights[var]);
	}
	free(c->pred.weights);
	free(c->pred.matrix);
	free(c->pred.pre);
}
     
void pred_update(CL *c, double *y, double *x)
{
	PRED *pred = &c->pred;
	int n = pred->weights_length;
	int n_sqrd = n*n;
	double tmp_input[n];
	double tmp_vec[n];
	double tmp_matrix1[n_sqrd];
	double tmp_matrix2[n_sqrd];

	tmp_input[0] = XCSF_X0;
	int index = 1;
	// linear terms
	for(int i = 0; i < num_x_vars; i++) {
		tmp_input[index++] = x[i];
	}
#if PRE == 3
	// quadratic terms
	for(int i = 0; i < num_x_vars; i++) {
		for(int j = i; j < num_x_vars; j++) {
			tmp_input[index++] = x[i] * x[j];
		}
	}
#endif

	// determine gain vector = matrix * tmp_input
	matrix_vector_multiply(pred->matrix, tmp_input, tmp_vec, n);

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
	for(int var = 0; var < num_y_vars; var++) {
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
	matrix_matrix_multiply(tmp_matrix1, pred->matrix, tmp_matrix2, n);

	// divide gain matrix entries by lambda
	for(int row = 0; row < n; row++) {
		for(int col = 0; col < n; col++) {
			pred->matrix[row*n+col] = tmp_matrix2[row*n+col] / RLS_LAMBDA;
		}
	}
}

double *pred_compute(CL *c, double *x)
{
	PRED *pred = &c->pred;
	for(int var = 0; var < num_y_vars; var++) {
		// first coefficient is offset
		double pre = XCSF_X0 * pred->weights[var][0];
		int index = 1;
		// multiply linear coefficients with the prediction input
		for(int i = 0; i < num_x_vars; i++) {
			pre += pred->weights[var][index++] * x[i];
		}
#if PRE == 3
		// multiply quadratic coefficients with prediction input
		for(int i = 0; i < num_x_vars; i++) {
			for(int j = i; j < num_x_vars; j++) {
				pre += pred->weights[var][index++] * x[i] * x[j];
			}
		}
#endif
		pred->pre[var] = pre;
	}
	return pred->pre;
} 

void pred_print(CL *c)
{
	PRED *pred = &c->pred;
	printf("RLS weights: ");
	for(int var = 0; var < num_y_vars; var++) {
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

void matrix_matrix_multiply(double *srca, double *srcb, double *dest, int n)
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
#endif
