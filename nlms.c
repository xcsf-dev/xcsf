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
 * The normalised least mean square classifier computed prediction module.
 *
 * Creates a weight vector representing a polynomial function to compute the
 * expected value given a problem instance and adapts the weights using the
 * least mean square update (also known as the modified Delta rule, or
 * Widrow-Hoff update.)
 */

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
	c->weights_length = (state_length*XCSF_EXPONENT)+1;
	c->weights = malloc(sizeof(double) * c->weights_length);
	for(int i = 0; i < c->weights_length; i++)
		c->weights[i] = 0.0;
}

void pred_copy(CL *to, CL *from)
{
	to->weights_length = from->weights_length;
	memcpy(to->weights, from->weights, sizeof(double)*from->weights_length);
}
 
void pred_free(CL *c)
{
	free(c->weights);
}

void pred_update(CL *c, double p, double *state)
{
	double error = p - pred_compute(c, state);
	double norm = XCSF_X0 * XCSF_X0;
	for(int i = 0; i < state_length; i++)
		norm += state[i] * state[i];
	double correction = (XCSF_ETA * error) / norm;
	c->weights[0] += XCSF_X0 * correction;
	for(int i = 0; i < c->weights_length-1; i+=XCSF_EXPONENT)
		for(int j = 0; j < XCSF_EXPONENT; j++)
			c->weights[i+j+1] += correction * pow(state[i/XCSF_EXPONENT], j+1);
}

double pred_compute(CL *c, double *state)
{
	double pre = XCSF_X0 * c->weights[0];
	for(int i = 0; i < c->weights_length-1; i+=XCSF_EXPONENT)
		for(int j = 0; j < XCSF_EXPONENT; j++)
			pre += pow(state[i/XCSF_EXPONENT], j+1) * c->weights[i+j+1];
	return pre;
} 
 

void pred_print(CL *c)
{
	printf("weights: ");
	for(int i = 0; i < c->weights_length; i++)
		printf("%f, ", c->weights[i]);
	printf("\n");
}
