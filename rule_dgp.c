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
 
#if CON == 11

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
	graph_init(&c->cond.dgp, DGP_NUM_NODES);
#ifdef SAM
	sam_init(&c->cond.mu);
#endif
}

void cond_free(CL *c)
{
	graph_free(&c->cond.dgp);
#ifdef SAM
	sam_free(c->cond.mu);
#endif
}

void cond_copy(CL *to, CL *from)
{
	graph_copy(&to->cond.dgp, &from->cond.dgp);
#ifdef SAM
	memcpy(to->cond.mu, from->cond.mu, sizeof(double)*NUM_MU);
#endif
}

void cond_rand(CL *c)
{
	graph_rand(&c->cond.dgp);
}

void cond_cover(CL *c, double *x)
{
	// generates random graphs until the network matches for input state
	do {
		cond_rand(c);
	} while(!cond_match(c, x));
}

_Bool cond_match(CL *c, double *x)
{
	// classifier matches if the first output node > 0.5
	graph_update(&c->cond.dgp, x);
	if(graph_output(&c->cond.dgp, 0) > 0.5) {
		c->cond.m = true;
	}
	else {
		c->cond.m = false;
	}
	return c->cond.m;
}

_Bool cond_mutate(CL *c)
{
	_Bool mod = false;
#ifdef SAM
	sam_adapt(c->cond.mu);
	if(NUM_MU > 0)
		P_MUTATION = c->cond.mu[0];
#endif

	mod = graph_mutate(&c->cond.dgp, P_MUTATION);
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
	graph_print(&c->cond.dgp);
}  

void pred_init(CL *c)
{
	c->pred.pre = malloc(sizeof(double)*num_y_vars);
}

void pred_free(CL *c)
{
	free(c->pred.pre);
}

void pred_copy(CL *to, CL *from)
{
	(void)to;
	(void)from;
}

void pred_update(CL *c, double *y, double *x)
{
	(void)c;
	(void)y;
	(void)x;
}

double *pred_compute(CL *c, double *x)
{
	(void)x;
	for(int i = 0; i < num_y_vars; i++) {
		c->pred.pre[i] = graph_output(&c->cond.dgp, 1+i);
	}
	return c->pred.pre;
}

void pred_print(CL *c)
{
	(void)c;
}

#endif
