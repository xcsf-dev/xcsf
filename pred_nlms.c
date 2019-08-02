/*
 * Copyright (C) 2015--2019 Richard Preen <rpreen@gmail.com>
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
	pred->weights_length = 1+2*state_length+state_length*(state_length-1)/2;
#else
	pred->weights_length = state_length+1;
#endif
	pred->weights = malloc(sizeof(double) * pred->weights_length);
	pred->weights[0] = XCSF_X0;
	for(int i = 1; i < pred->weights_length; i++)
		pred->weights[i] = 0.0;
}

void pred_copy(CL *to, CL *from)
{
	memcpy(to->pred.weights, from->pred.weights, sizeof(double)*from->pred.weights_length);
}

void pred_free(CL *c)
{
	free(c->pred.weights);
}

void pred_update(CL *c, double p, double *state)
{
	PRED *pred = &c->pred;
	// pre has been updated for the current state during set_pred()
	double error = p - pred->pre; //pred_compute(c, state);
	double norm = XCSF_X0 * XCSF_X0;
	for(int i = 0; i < state_length; i++)
		norm += state[i] * state[i];
	double correction = (XCSF_ETA * error) / norm;
	// update first coefficient
	pred->weights[0] += XCSF_X0 * correction;
	int index = 1;
	// update linear coefficients
	for(int i = 0; i < state_length; i++)
		pred->weights[index++] += correction * state[i];
#if PRE == 1
	// update quadratic coefficients
	for(int i = 0; i < state_length; i++) {
		for(int j = i; j < state_length; j++) {
			pred->weights[index++] += correction * state[i] * state[j];
		}
	}
#endif
}

double pred_compute(CL *c, double *state)
{
	PRED *pred = &c->pred;
	// first coefficient is offset
	double pre = XCSF_X0 * pred->weights[0];
	int index = 1;
	// multiply linear coefficients with the prediction input
	for(int i = 0; i < state_length; i++)
		pre += pred->weights[index++] * state[i];
#if PRE == 1
	// multiply quadratic coefficients with prediction input
	for(int i = 0; i < state_length; i++) {
		for(int j = i; j < state_length; j++) {
			pre += pred->weights[index++] * state[i] * state[j];
		}
	}
#endif
	pred->pre = pre;
	return pre;
} 

void pred_print(CL *c)
{
	PRED *pred = &c->pred;
	printf("weights: ");
	for(int i = 0; i < pred->weights_length; i++)
		printf("%f, ", pred->weights[i]);
	printf("\n");
}
#endif
