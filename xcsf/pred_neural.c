/*
 * Copyright (C) 2016--2019 Richard Preen <rpreen@gmail.com>
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
	int neurons[3] = {num_x_vars, NUM_HIDDEN_NEURONS, num_y_vars};
	double (*activations[2])(double) = {sig, sig};
	neural_init(&c->pred.bpn, 3, neurons, activations);
	c->pred.pre = malloc(sizeof(double)*num_y_vars);
}

void pred_free(CL *c)
{
	neural_free(&c->pred.bpn);
	free(c->pred.pre);
}

void pred_copy(CL *to, CL *from)
{
	neural_copy(&to->pred.bpn, &from->pred.bpn);
}

void pred_update(CL *c, double *y, double *x)
{
	neural_learn(&c->pred.bpn, y, x);
}

double *pred_compute(CL *c, double *x)
{
	neural_propagate(&c->pred.bpn, x);
	for(int i = 0; i < num_y_vars; i++) {
		c->pred.pre[i] = neural_output(&c->pred.bpn, i);
	}
	return c->pred.pre;
}

void pred_print(CL *c)
{
	neural_print(&c->pred.bpn);
}  

#endif
