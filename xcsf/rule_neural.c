/*
 * Copyright (C) 2019 Richard Preen <rpreen@gmail.com>
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
 * The neural classifier rule module.
 * Performs both condition matching and prediction.
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
#include "rule_neural.h"
  
typedef struct RULE_NEURAL_COND {
	BPN bpn;
	_Bool m;
	double *mu;
} RULE_NEURAL_COND;
       
typedef struct RULE_NEURAL_PRED {
	double *pre;
} RULE_NEURAL_PRED;
 
void rule_neural_cond_init(CL *c)
{
	RULE_NEURAL_COND *cond = malloc(sizeof(RULE_NEURAL_COND));
	int neurons[3] = {num_x_vars, NUM_HIDDEN_NEURONS, num_y_vars+1};
	double (*activations[2])(double) = {sig, sig};
	neural_init(&cond->bpn, 3, neurons, activations);
	c->cond = cond;
	sam_init(&cond->mu);
}

void rule_neural_cond_free(CL *c)
{
	RULE_NEURAL_COND *cond = c->cond;
	neural_free(&cond->bpn);
	sam_free(cond->mu);
	free(c->cond);
}  

double rule_neural_cond_mu(CL *c, int m)
{
	RULE_NEURAL_COND *cond = c->cond;
	return cond->mu[m];
}

void rule_neural_cond_copy(CL *to, CL *from)
{
	RULE_NEURAL_COND *to_cond = to->cond;
	RULE_NEURAL_COND *from_cond = from->cond;
	neural_copy(&to_cond->bpn, &from_cond->bpn);
	memcpy(to_cond->mu, from_cond->mu, sizeof(double)*NUM_MU);
}

void rule_neural_cond_rand(CL *c)
{
	RULE_NEURAL_COND *cond = c->cond;
	neural_rand(&cond->bpn);
}

void rule_neural_cond_cover(CL *c, double *x)
{
	// generates random weights until the network matches for input state
	do {
		rule_neural_cond_rand(c);
	} while(!rule_neural_cond_match(c, x));
}

_Bool rule_neural_cond_match(CL *c, double *x)
{
	// classifier matches if the first output neuron > 0.5
	RULE_NEURAL_COND *cond = c->cond;
	neural_propagate(&cond->bpn, x);
	if(neural_output(&cond->bpn, 0) > 0.5) {
		cond->m = true;
	}
	else {
		cond->m = false;
	}
	return cond->m;
}    

_Bool rule_neural_cond_match_state(CL *c)
{
	RULE_NEURAL_COND *cond = c->cond;
	return cond->m;
}

_Bool rule_neural_cond_mutate(CL *c)
{
	RULE_NEURAL_COND *cond = c->cond;
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

_Bool rule_neural_cond_crossover(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}

_Bool rule_neural_cond_subsumes(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}

_Bool rule_neural_cond_general(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}   

void rule_neural_cond_print(CL *c)
{
	RULE_NEURAL_COND *cond = c->cond;
	neural_print(&cond->bpn);
}  
 
void rule_neural_pred_init(CL *c)
{
	RULE_NEURAL_PRED *pred = malloc(sizeof(RULE_NEURAL_COND));
	pred->pre = malloc(sizeof(double)*num_y_vars);
	c->pred = pred;
}

void rule_neural_pred_free(CL *c)
{
	RULE_NEURAL_PRED *pred = c->pred;
	free(pred->pre);
	free(c->pred);
}

void rule_neural_pred_copy(CL *to, CL *from)
{
	(void)to;
	(void)from;
}

void rule_neural_pred_update(CL *c, double *y, double *x)
{
	(void)c;
	(void)y;
	(void)x;
}

double *rule_neural_pred_compute(CL *c, double *x)
{
	(void)x;
	RULE_NEURAL_COND *cond = c->cond;
	RULE_NEURAL_PRED *pred = c->pred;
	for(int i = 0; i < num_y_vars; i++) {
		pred->pre[i] = neural_output(&cond->bpn, 1+i);
	}
	return pred->pre;
}
  
double rule_neural_pred_pre(CL *c, int p)
{
	RULE_NEURAL_PRED *pred = c->pred;
	return pred->pre[p];
}
 
void rule_neural_pred_print(CL *c)
{
	(void)c;
}  
