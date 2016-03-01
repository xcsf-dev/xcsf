/*
 * Copyright (C) 2016 Riintd Preen <rpreen@gmail.com>
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
 * The tree GP condition module.
 *
 * Provides functionality to create GP trees that compute whether the
 * classifier matches for a given problem instance. Includes operations for
 * covering, matching, copying, mutating, printing, etc.
 */
  
#if CON == 2

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
	tree_init(&cond->gp);
#ifdef SAM
	sam_init(&cond->mu);
#endif
}

void cond_free(COND *cond)
{
	tree_free(&cond->gp);
#ifdef SAM
	sam_free(cond->mu);
#endif
}

void cond_copy(COND *to, COND *from)
{
	tree_copy(&to->gp, &from->gp);
#ifdef SAM
	memcpy(to->mu, from->mu, sizeof(double)*NUM_MU);
#endif
}

void cond_rand(COND *cond)
{
	tree_free(&cond->gp);
	tree_rand(&cond->gp);
}

void cond_cover(COND *cond, double *state)
{
	// generates random weights until the tree matches for input state
	do {
		cond_rand(cond);
	} while(!cond_match(cond, state));
}

_Bool cond_match(COND *cond, double *state)
{
	// classifier matches if the tree output > 0.5
	cond->gp.p = 0;
	double result = tree_eval(&cond->gp, state);
	if(result > 0.5) {
		cond->m = true;
		return true;
	}
	cond->m = false;
	return false;
}

_Bool cond_mutate(COND *cond)
{
#ifdef SAM
	sam_adapt(cond->mu);
	if(NUM_MU > 0) {
		P_MUTATION = cond->mu[0];
		if(NUM_MU > 1)
			S_MUTATION = cond->mu[1];
	}
#endif

	if(drand() < P_MUTATION) {
		tree_mutation(&cond->gp, P_MUTATION);
		return true;
	}
	else {
		return false;
	}
}

_Bool cond_crossover(COND *cond1, COND *cond2)
{
	if(drand() < P_CROSSOVER) {
		tree_crossover(&cond1->gp, &cond2->gp);
		return true;
	}
	else {
		return false;
	}
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
	tree_print(&cond->gp, 0);
}  

#endif
