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
 * The tree GP condition module.
 *
 * Provides functionality to create GP trees that compute whether the
 * classifier matches for a given problem instance. Includes operations for
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
#include "cond_gp.h"

void cond_gp_init(CL *c)
{
	COND_GP *cond = malloc(sizeof(COND_GP));
	tree_init(&cond->gp);
	c->cond = cond;
	sam_init(&cond->mu);
}

void cond_gp_free(CL *c)
{
	COND_GP *cond = c->cond;
	tree_free(&cond->gp);
	sam_free(cond->mu);
	free(c->cond);
}
 
double cond_gp_mu(CL *c, int m)
{
	COND_GP *cond = c->cond;
	return cond->mu[m];
}
 
void cond_gp_copy(CL *to, CL *from)
{
	COND_GP *to_cond = to->cond;
	COND_GP *from_cond = from->cond;
	tree_copy(&to_cond->gp, &from_cond->gp);
	memcpy(to_cond->mu, from_cond->mu, sizeof(double)*NUM_MU);
}

void cond_gp_rand(CL *c)
{
	COND_GP *cond = c->cond;
	tree_free(&cond->gp);
	tree_rand(&cond->gp);
}

void cond_gp_cover(CL *c, double *state)
{
	// generates random weights until the tree matches for input state
	do {
		cond_gp_rand(c);
	} while(!cond_gp_match(c, state));
}

_Bool cond_gp_match(CL *c, double *state)
{
	// classifier matches if the tree output > 0.5
	COND_GP *cond = c->cond;
	cond->gp.p = 0;
	double result = tree_eval(&cond->gp, state);
	if(result > 0.5) {
		cond->m = true;
	}
	else {
		cond->m = false;
	}
	return cond->m;
}    

_Bool cond_gp_match_state(CL *c)
{
	COND_GP *cond = c->cond;
	return cond->m;
}
 
_Bool cond_gp_mutate(CL *c)
{
	COND_GP *cond = c->cond;
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

_Bool cond_gp_crossover(CL *c1, CL *c2)
{
	COND_GP *cond1 = c1->cond;
	COND_GP *cond2 = c2->cond;
	if(drand() < P_CROSSOVER) {
		tree_crossover(&cond1->gp, &cond2->gp);
		return true;
	}
	else {
		return false;
	}
}

_Bool cond_gp_subsumes(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}

_Bool cond_gp_general(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}   

void cond_gp_print(CL *c)
{
	COND_GP *cond = c->cond;
	tree_print(&cond->gp, 0);
}  
