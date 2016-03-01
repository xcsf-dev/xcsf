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
 */
 
#if CON == 3

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
	graph_init(&cond->dgp, DGP_NUM_NODES);
#ifdef SAM
	sam_init(&cond->mu);
#endif
}

void cond_free(COND *cond)
{
	graph_free(&cond->dgp);
#ifdef SAM
	sam_free(cond->mu);
#endif
}

void cond_copy(COND *to, COND *from)
{
	graph_copy(&to->dgp, &from->dgp);
#ifdef SAM
	memcpy(to->mu, from->mu, sizeof(double)*NUM_MU);
#endif
}

void cond_rand(COND *cond)
{
	graph_rand(&cond->dgp);
}

void cond_cover(COND *cond, double *state)
{
	// generates random graphs until the network matches for input state
	do {
		cond_rand(cond);
	} while(!cond_match(cond, state));
}

_Bool cond_match(COND *cond, double *state)
{
	// classifier matches if the first output node > 0.5
	graph_update(&cond->dgp, state);
	if(graph_output(&cond->dgp, 0) > 0.5) {
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
	if(NUM_MU > 0)
		P_MUTATION = cond->mu[0];
#endif

	mod = graph_mutate(&cond->dgp, P_MUTATION);
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
	graph_print(&cond->dgp);
}  

#endif
