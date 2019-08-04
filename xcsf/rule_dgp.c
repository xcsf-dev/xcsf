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
 * The DGP classifier rule module.
 * Performs both condition matching and prediction in a single evolved graph.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "cl.h"
#include "dgp.h"
#include "rule_dgp.h"
  
typedef struct RULE_DGP_COND {
	GRAPH dgp;
	_Bool m;
	double *mu;
} RULE_DGP_COND;
  
typedef struct RULE_DGP_PRED {
	double *pre;
} RULE_DGP_PRED;
 
void rule_dgp_cond_init(CL *c)
{
	RULE_DGP_COND *cond = malloc(sizeof(RULE_DGP_COND));
	graph_init(&cond->dgp, DGP_NUM_NODES);
	c->cond = cond;
	sam_init(&cond->mu);
}

void rule_dgp_cond_free(CL *c)
{
	RULE_DGP_COND *cond = c->cond;
	graph_free(&cond->dgp);
	sam_free(cond->mu);
	free(c->cond);
}
 
double rule_dgp_cond_mu(CL *c, int m)
{
	RULE_DGP_COND *cond = c->cond;
	return cond->mu[m];
}
 
void rule_dgp_cond_copy(CL *to, CL *from)
{
	RULE_DGP_COND *to_cond = to->cond;
	RULE_DGP_COND *from_cond = from->cond;
	graph_copy(&to_cond->dgp, &from_cond->dgp);
	sam_copy(to_cond->mu, from_cond->mu);
}

void rule_dgp_cond_rand(CL *c)
{
	RULE_DGP_COND *cond = c->cond;
	graph_rand(&cond->dgp);
}

void rule_dgp_cond_cover(CL *c, double *x)
{
	// generates random graphs until the network matches for input state
	do {
		cond_rand(c);
	} while(!cond_match(c, x));
}

_Bool rule_dgp_cond_match(CL *c, double *x)
{
	// classifier matches if the first output node > 0.5
	RULE_DGP_COND *cond = c->cond;
	graph_update(&cond->dgp, x);
	if(graph_output(&cond->dgp, 0) > 0.5) {
		cond->m = true;
	}
	else {
		cond->m = false;
	}
	return cond->m;
}    

_Bool rule_dgp_cond_match_state(CL *c)
{
	RULE_DGP_COND *cond = c->cond;
	return cond->m;
}

_Bool rule_dgp_cond_mutate(CL *c)
{
	RULE_DGP_COND *cond = c->cond;
	_Bool mod = false;
	if(NUM_SAM > 0) {
		sam_adapt(cond->mu);
		P_MUTATION = cond->mu[0];
	}
	mod = graph_mutate(&cond->dgp, P_MUTATION);
	return mod;
}

_Bool rule_dgp_cond_crossover(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}

_Bool rule_dgp_cond_subsumes(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}

_Bool rule_dgp_cond_general(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}   

void rule_dgp_cond_print(CL *c)
{
	RULE_DGP_COND *cond = c->cond;
	graph_print(&cond->dgp);
}  

void rule_dgp_pred_init(CL *c)
{
	RULE_DGP_PRED *pred = malloc(sizeof(RULE_DGP_PRED));
	pred->pre = malloc(sizeof(double)*num_y_vars);
	c->pred = pred;
}

void rule_dgp_pred_free(CL *c)
{
	RULE_DGP_PRED *pred = c->pred;
	free(pred->pre);
	free(pred);
}

void rule_dgp_pred_copy(CL *to, CL *from)
{
	(void)to;
	(void)from;
}

void rule_dgp_pred_update(CL *c, double *y, double *x)
{
	(void)c;
	(void)y;
	(void)x;
}

double *rule_dgp_pred_compute(CL *c, double *x)
{
	(void)x;
	RULE_DGP_COND *cond = c->cond;
	RULE_DGP_PRED *pred = c->pred;
	for(int i = 0; i < num_y_vars; i++) {
		pred->pre[i] = graph_output(&cond->dgp, 1+i);
	}
	return pred->pre;
}
 
double rule_dgp_pred_pre(CL *c, int p)
{
	RULE_DGP_PRED *pred = c->pred;
	return pred->pre[p];
}
 
void rule_dgp_pred_print(CL *c)
{
	(void)c;
}
