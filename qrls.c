/*
 * Copyright (C) 2015 Richard Preen <rpreen@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
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
 * The quadratic recursive least mean square classifier computed prediction
 * module.
 *
 */

#ifdef RLS
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
void init_matrix(double *matrix);

double *tmpExtendedPredInput;
double *tmpGainVector;
double *tmpMatrix1;
double *tmpMatrix2;

void pred_init(CL *c)
{
	// offset(1) + n linear + n quadratic + n*(n-1)/2 mixed terms
	c->weights_length = 1+2*state_length+state_length*(state_length-1)/2;
	c->weights = malloc(sizeof(double) * c->weights_length);
	c->weights[0] = XCSF_X0;
	for(int i = 1; i < c->weights_length; i++)
		c->weights[i] = 0.0;

	// initialise gain matrix
	c->matrix = malloc(sizeof(double)*pow(c->weights_length,2));
	init_matrix(c->matrix);

	// initialise temporary arrays (only needs to be done once)
	if(tmpGainVector == NULL) {
		tmpGainVector = malloc(sizeof(double)*c->weights_length);
		tmpExtendedPredInput = malloc(sizeof(double)*c->weights_length);
		tmpMatrix1 = malloc(sizeof(double)*pow(c->weights_length,2));
		tmpMatrix2 = malloc(sizeof(double)*pow(c->weights_length,2));
	}
}
 	
void init_matrix(double *matrix)
{
	for(int row = 0; row < state_length; row++) {
		for(int col = 0; col < state_length; col++) {
			if(row != col)
				matrix[row*state_length+col] = 0.0;
			else
				matrix[row*state_length+col] = RLS_SCALE_FACTOR;
		}
	}
}
 
void pred_copy(CL *to, CL *from)
{
	to->weights_length = from->weights_length;
	memcpy(to->weights, from->weights, sizeof(double)*from->weights_length);
	memcpy(to->matrix, from->matrix, sizeof(double)*pow(from->weights_length,2));
	init_matrix(to->matrix);
}
 
void pred_free(CL *c)
{
	free(c->weights);
	free(c->matrix);
}

void pred_update(CL *c, double p, double *state)
{
	tmpExtendedPredInput[0] = XCSF_X0;
	int index = 1;
	// linear terms
	for(int i = 0; i < state_length; i++)
		tmpExtendedPredInput[index++] = state[i];
	// quadratic terms
	for(int i = 0; i < state_length; i++)
		for(int j = 0; j < state_length; j++)
			tmpExtendedPredInput[index++] = pow(state[i],2);


	// 1. determine gain vector = matrix * extendedPredInput
	matrix_vector_multiply(c->matrix, tmpExtendedPredInput, tmpGainVector, c->weights_length);
	
	// 2. divide gain vector by lambda + k
	double divisor = RLS_LAMBDA;
	for(int i = 0; i < c->weights_length; i++)
		divisor += tmpExtendedPredInput[i] * tmpGainVector[i];
	for(int i = 0; i < c->weights_length; i++)
		tmpGainVector[i] /= divisor;

	// 3. update weights using the error
	double error = p - pred_compute(c, state);
	for(int i = 0; i < c->weights_length; i++)
		c->weights[i] += error * tmpGainVector[i];

	// 4. update gain matrix
	for(int i = 0; i < c->weights_length; i++) {
		for(int j = 0; j < c->weights_length; j++) {
			double tmp = tmpGainVector[i] * tmpExtendedPredInput[j];
			if(i == j)
				tmpMatrix1[i*state_length+j] = 1.0 - tmp;
			else
				tmpMatrix1[i*state_length+j] = -tmp;
		}
	}
	matrix_matrix_multiply(tmpMatrix1, c->matrix, tmpMatrix2, c->weights_length);

	// 5. divide gain matrix entries by lambda
	for(int row = 0; row < c->weights_length; row++) {
		for(int col = 0; col < c->weights_length; col++) {
			c->matrix[row*state_length+col] = tmpMatrix2[row*state_length+col] / RLS_LAMBDA;
		}
	}
}

double pred_compute(CL *c, double *state)
{
	// first coefficient is offset
	double pre = XCSF_X0 * c->weights[0];
	int index = 1;
	// multiply linear coefficients with the prediction input
	for(int i = 0; i < state_length; i++)
		pre += c->weights[index++] * state[i];
	// multiply quadratic coefficients with prediction input
	for(int i = 0; i < state_length; i++) {
		for(int j = i; j < state_length; j++) {
			pre += c->weights[index++] * state[i] * state[j];
		}
	}
	return pre;
} 


void pred_print(CL *c)
{
	printf("weights: ");
	for(int i = 0; i < c->weights_length; i++)
		printf("%f, ", c->weights[i]);
	printf("\n");
}

void matrix_matrix_multiply(double *srca, double *srcb, double *dest, int n)
{
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			dest[i*state_length+j] = srca[i*state_length] * srcb[j];
			for(int k = 0; k < n; k++) {
				dest[i*state_length+j] += srca[i*state_length+k] * srcb[k*state_length+j];
			}
		}
	}
}

void matrix_vector_multiply(double *srcm, double *srcv, double *dest, int n)
{
	for(int i = 0; i < n; i++) {
		dest[i] = srcm[i*state_length] * srcv[0];
		for(int j = 1; j < n; j++) {
			dest[i] += srcm[i*state_length+j] * srcv[j];
		}
	}
}
#endif
