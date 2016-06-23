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

void cond_init(COND *cond)
{
	int neurons[3] = {state_length, NUM_HIDDEN_NEURONS, 1};
	neural_init(&cond->bpn, 3, neurons);
#ifdef SAM
	sam_init(&cond->mu);
#endif
}

void cond_free(COND *cond)
{
	neural_free(&cond->bpn);
#ifdef SAM
	sam_free(cond->mu);
#endif
}

void cond_copy(COND *to, COND *from)
{
	neural_copy(&to->bpn, &from->bpn);
#ifdef SAM
	memcpy(to->mu, from->mu, sizeof(double)*NUM_MU);
#endif
}

void cond_rand(COND *cond)
{
	neural_rand(&cond->bpn);
}

void cond_cover(COND *cond, double *state)
{
	// generates random weights until the network matches for input state
	do {
		cond_rand(cond);
	} while(!cond_match(cond, state));
}

_Bool cond_match(COND *cond, double *state)
{
	// classifier matches if the first output neuron > 0.5
	neural_propagate(&cond->bpn, state);
	if(neural_output(&cond->bpn, 0) > 0.5) {
		cond->m = true;
		return true;
	}
	cond->m = false;
	return false;
}

_Bool cond_mutate(COND *cond)
{
	_Bool mod = false;
#ifdef SAM
	sam_adapt(cond->mu);
	if(NUM_MU > 0) {
		P_MUTATION = cond->mu[0];
		if(NUM_MU > 1)
			S_MUTATION = cond->mu[1];
	}
#endif
	BPN *bpn = &cond->bpn;
	for(int l = 1; l < bpn->num_layers; l++) {
		for(int i = 0; i < bpn->num_neurons[l]; i++) {
			NEURON *n = &bpn->layer[l-1][i];
			for(int w = 0; w < n->num_inputs+1; w++) {
				if(drand() < P_MUTATION) {
					double orig = n->weights[w];
					n->weights[w] += ((drand()*2.0)-1.0)*S_MUTATION;
//					if(n->weights[w] > 1.0)
//						n->weights[w] = 1.0;
//					else if(n->weights[w] < -1.0)
//						n->weights[w] = -1.0;
					if(n->weights[w] != orig)
						mod = true;
				}
			}
		}
	}
	return mod;
}

_Bool cond_crossover(COND *cond1, COND *cond2)
{
	// remove unused parameter warnings
	(void)cond1;
	(void)cond2;
	return false;
}

_Bool cond_subsumes(COND *cond1, COND *cond2)
{
	// remove unused parameter warnings
	(void)cond1;
	(void)cond2;
	return false;
}

_Bool cond_general(COND *cond1, COND *cond2)
{
	// remove unused parameter warnings
	(void)cond1;
	(void)cond2;
	return false;
}   

void cond_print(COND *cond)
{
	neural_print(&cond->bpn);
}  

#endif
