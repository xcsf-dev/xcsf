/*
 * Copyright (C) 2012--2015 Richard Preen <rpreen@gmail.com>
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

#ifdef NEURAL_PREDICTION

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "cl.h"

void pred_init(PRED *pred)
{
	neural_init(&pred->bpn);
}

void pred_free(PRED *pred)
{
	neural_free(&pred->bpn);
}

void pred_copy(PRED *to, PRED *from)
{
	neural_copy(&to->bpn, &from->bpn);
}

double out[1];
void pred_update(PRED *pred, double p, double *state)
{
	out[0] = p;
	neural_learn(&pred->bpn, out, state);
}

double pred_compute(PRED *pred, double *state)
{
	neural_propagate(&pred->bpn, state);
	pred->pre = neural_output(&pred->bpn, 0);
	return pred->pre;
}

void pred_print(PRED *pred)
{
	neural_print(&pred->bpn);
}  

#endif
