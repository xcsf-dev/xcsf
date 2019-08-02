/*
 * Copyright (C) 2016 Richard Preen <rpreen@gmail.com>
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
 * The MLP neural network classifier computed prediction module.
 *
 * Creates a weight vector representing an MLP neural network to calculate the
 * expected value given a problem instance and adapts the weights using the
 * backpropagation algorithm.
 */

#if PRE == 4

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "cl.h"

void pred_init(CL *c)
{
	int neurons[3] = {state_length, NUM_HIDDEN_NEURONS, 1};
	double (*activations[2])(double) = {sig, sig};
	neural_init(&c->pred.bpn, 3, neurons, activations);
}

void pred_free(CL *c)
{
	neural_free(&c->pred.bpn);
}

void pred_copy(CL *to, CL *from)
{
	neural_copy(&to->pred.bpn, &from->pred.bpn);
}

void pred_update(CL *c, double p, double *state)
{
	double out[1];
	out[0] = p;
	neural_learn(&c->pred.bpn, out, state);
}

double pred_compute(CL *c, double *state)
{
	neural_propagate(&c->pred.bpn, state);
	c->pred.pre = neural_output(&c->pred.bpn, 0);
	return c->pred.pre;
}

void pred_print(CL *c)
{
	neural_print(&c->pred.bpn);
}  

#endif
