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
#include "cond_dgp.h"
  
typedef struct COND_DGP {
	GRAPH dgp;
	_Bool m;
	double *mu;
} COND_DGP;
 
void cond_dgp_init(CL *c)
{
	COND_DGP *cond = malloc(sizeof(COND_DGP));
	graph_init(&cond->dgp, DGP_NUM_NODES);
	c->cond = cond;
	sam_init(&cond->mu);
}

void cond_dgp_free(CL *c)
{
	COND_DGP *cond = c->cond;
	graph_free(&cond->dgp);
	sam_free(cond->mu);
	free(c->cond);
}                  

double cond_dgp_mu(CL *c, int m)
{
	COND_DGP *cond = c->cond;
	return cond->mu[m];
}
 
void cond_dgp_copy(CL *to, CL *from)
{
	COND_DGP *to_cond = to->cond;
	COND_DGP *from_cond = from->cond;
	graph_copy(&to_cond->dgp, &from_cond->dgp);
	memcpy(to_cond->mu, from_cond->mu, sizeof(double)*NUM_MU);
}

void cond_dgp_rand(CL *c)
{
	COND_DGP *cond = c->cond;
	graph_rand(&cond->dgp);
}

void cond_dgp_cover(CL *c, double *state)
{
	// generates random graphs until the network matches for input state
	do {
		cond_dgp_rand(c);
	} while(!cond_dgp_match(c, state));
}

_Bool cond_dgp_match(CL *c, double *state)
{
	// classifier matches if the first output node > 0.5
	COND_DGP *cond = c->cond;
	graph_update(&cond->dgp, state);
	if(graph_output(&cond->dgp, 0) > 0.5) {
		cond->m = true;
	}
	else {
		cond->m = false;
	}
	return cond->m;
}            

_Bool cond_dgp_match_state(CL *c)
{
	COND_DGP *cond = c->cond;
	return cond->m;
}

_Bool cond_dgp_mutate(CL *c)
{
	COND_DGP *cond = c->cond;
	_Bool mod = false;
#ifdef SAM
	sam_adapt(cond->mu);
	if(NUM_MU > 0)
		P_MUTATION = cond->mu[0];
#endif

	mod = graph_mutate(&cond->dgp, P_MUTATION);
	return mod;
}

_Bool cond_dgp_crossover(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}

_Bool cond_dgp_subsumes(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}

_Bool cond_dgp_general(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}   

void cond_dgp_print(CL *c)
{
	COND_DGP *cond = c->cond;
	graph_print(&cond->dgp);
}  
