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
 * The neural classifier condition module.
 *
 * Provides functionality to create MLP neural networks that compute whether
 * the classifier matches for a given problem instance. Includes operations for
 * covering, matching, copying, mutating, printing, etc.
 */

#if CON == 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "cl.h"

void cond_init(CL *c)
{
	int neurons[3] = {num_x_vars, NUM_HIDDEN_NEURONS, 1};
	double (*activations[2])(double) = {sig, sig};
	neural_init(&c->cond.bpn, 3, neurons, activations);
#ifdef SAM
	sam_init(&c->cond.mu);
#endif
}

void cond_free(CL *c)
{
	neural_free(&c->cond.bpn);
#ifdef SAM
	sam_free(c->cond.mu);
#endif
}

void cond_copy(CL *to, CL *from)
{
	neural_copy(&to->cond.bpn, &from->cond.bpn);
#ifdef SAM
	memcpy(to->cond.mu, from->cond.mu, sizeof(double)*NUM_MU);
#endif
}

void cond_rand(CL *c)
{
	neural_rand(&c->cond.bpn);
}

void cond_cover(CL *c, double *x)
{
	// generates random weights until the network matches for input state
	do {
		cond_rand(c);
	} while(!cond_match(c, x));
}

_Bool cond_match(CL *c, double *x)
{
	// classifier matches if the first output neuron > 0.5
	neural_propagate(&c->cond.bpn, x);
	if(neural_output(&c->cond.bpn, 0) > 0.5) {
		c->cond.m = true;
	}
	else {
		c->cond.m = false;
	}
	return c->cond.m;
}

_Bool cond_mutate(CL *c)
{
	COND *cond = &c->cond;
	_Bool mod = false;
#ifdef SAM
	sam_adapt(cond->mu);
	if(NUM_MU > 0) {
		S_MUTATION = cond->mu[0];
	}
#endif
	BPN *bpn = &cond->bpn;
	for(int l = 1; l < bpn->num_layers; l++) {
		for(int i = 0; i < bpn->num_neurons[l]; i++) {
			NEURON *n = &bpn->layer[l-1][i];
			for(int w = 0; w < n->num_inputs+1; w++) {
				double orig = n->weights[w];
				n->weights[w] += ((drand()*2.0)-1.0)*S_MUTATION;
				if(n->weights[w] != orig)
					mod = true;
			}
		}
	}
	return mod;
}

_Bool cond_crossover(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}

_Bool cond_subsumes(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}

_Bool cond_general(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}   

void cond_print(CL *c)
{
	neural_print(&c->cond.bpn);
}  

#endif
