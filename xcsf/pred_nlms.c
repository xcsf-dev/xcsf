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
 * The normalised least mean square classifier computed prediction module.
 *
 * Creates a weight vector representing a polynomial function to compute the
 * expected value given a problem instance and adapts the weights using the
 * least mean square update (also known as the modified Delta rule, or
 * Widrow-Hoff update.)
 */

#if PRE == 0 || PRE == 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "cl.h"

void pred_init(CL *c)
{
	PRED *pred = &c->pred;
#if PRE == 1
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

	pred->pre = malloc(sizeof(double)*num_y_vars);
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
	free(c->pred.pre);
}

void pred_update(CL *c, double *y, double *x)
{
	PRED *pred = &c->pred;

	double norm = XCSF_X0 * XCSF_X0;
	for(int i = 0; i < num_x_vars; i++) {
		norm += x[i] * x[i];
	}      

	// pre has been updated for the current state during set_pred()
	for(int var = 0; var < num_y_vars; var++) {
		double error = y[var] - pred->pre[var]; // pred_compute(c, x);
		double correction = (XCSF_ETA * error) / norm;
		// update first coefficient
		pred->weights[var][0] += XCSF_X0 * correction;
		int index = 1;
		// update linear coefficients
		for(int i = 0; i < num_x_vars; i++) {
			pred->weights[var][index++] += correction * x[i];
		}
#if PRE == 1
		// update quadratic coefficients
		for(int i = 0; i < num_x_vars; i++) {
			for(int j = i; j < num_x_vars; j++) {
				pred->weights[var][index++] += correction * x[i] * x[j];
			}
		}
#endif
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
#if PRE == 1
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
	printf("weights: ");
	for(int var = 0; var < num_y_vars; var++) {
		for(int i = 0; i < pred->weights_length; i++) {
			printf("%f, ", pred->weights[var][i]);
		}
		printf("\n");
	}
}
#endif
