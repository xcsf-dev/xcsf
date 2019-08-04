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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "cl.h"
#include "neural.h"
#include "cond_neural.h"
 
typedef struct COND_NEURAL {
	BPN bpn;
	_Bool m;
	double *mu;
} COND_NEURAL;
 
void cond_neural_init(CL *c)
{
	COND_NEURAL *cond = malloc(sizeof(COND_NEURAL));
	int neurons[3] = {num_x_vars, NUM_HIDDEN_NEURONS, 1};
	double (*activations[2])(double) = {sig, sig};
	neural_init(&cond->bpn, 3, neurons, activations);
	c->cond = cond;
	sam_init(&cond->mu);
}

void cond_neural_free(CL *c)
{
	COND_NEURAL *cond = c->cond;
	neural_free(&cond->bpn);
	sam_free(cond->mu);
	free(c->cond);
}                  

double cond_neural_mu(CL *c, int m)
{
	COND_NEURAL *cond = c->cond;
	return cond->mu[m];
}
 
void cond_neural_copy(CL *to, CL *from)
{
	COND_NEURAL *to_cond = to->cond;
	COND_NEURAL *from_cond = from->cond;
	neural_copy(&to_cond->bpn, &from_cond->bpn);
	sam_copy(to_cond->mu, from_cond->mu);
}

void cond_neural_rand(CL *c)
{
	COND_NEURAL *cond = c->cond;
	neural_rand(&cond->bpn);
}

void cond_neural_cover(CL *c, double *x)
{
	// generates random weights until the network matches for input state
	do {
		cond_neural_rand(c);
	} while(!cond_neural_match(c, x));
}

_Bool cond_neural_match(CL *c, double *x)
{
	// classifier matches if the first output neuron > 0.5
	COND_NEURAL *cond = c->cond;
	neural_propagate(&cond->bpn, x);
	if(neural_output(&cond->bpn, 0) > 0.5) {
		cond->m = true;
	}
	else {
		cond->m = false;
	}
	return cond->m;
}                

_Bool cond_neural_match_state(CL *c)
{
	COND_NEURAL *cond = c->cond;
	return cond->m;
}

_Bool cond_neural_mutate(CL *c)
{
	COND_NEURAL *cond = c->cond;
	_Bool mod = false;
	if(NUM_SAM > 0) {
		sam_adapt(cond->mu);
		S_MUTATION = cond->mu[0];
	}
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

_Bool cond_neural_crossover(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}

_Bool cond_neural_subsumes(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}

_Bool cond_neural_general(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}   

void cond_neural_print(CL *c)
{
	COND_NEURAL *cond = c->cond;
	neural_print(&cond->bpn);
}  
