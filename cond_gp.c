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
  
#if CON == 2

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
	tree_init(&c->cond.gp);
#ifdef SAM
	sam_init(&c->cond.mu);
#endif
}

void cond_free(CL *c)
{
	tree_free(&c->cond.gp);
#ifdef SAM
	sam_free(c->cond.mu);
#endif
}

void cond_copy(CL *to, CL *from)
{
	tree_copy(&to->cond.gp, &from->cond.gp);
#ifdef SAM
	memcpy(to->cond.mu, from->cond.mu, sizeof(double)*NUM_MU);
#endif
}

void cond_rand(CL *c)
{
	tree_free(&c->cond.gp);
	tree_rand(&c->cond.gp);
}

void cond_cover(CL *c, double *state)
{
	// generates random weights until the tree matches for input state
	do {
		cond_rand(c);
	} while(!cond_match(c, state));
}

_Bool cond_match(CL *c, double *state)
{
	// classifier matches if the tree output > 0.5
	c->cond.gp.p = 0;
	double result = tree_eval(&c->cond.gp, state);
	if(result > 0.5) {
		c->cond.m = true;
		return true;
	}
	c->cond.m = false;
	return false;
}

_Bool cond_mutate(CL *c)
{
#ifdef SAM
	sam_adapt(c->cond.mu);
	if(NUM_MU > 0) {
		P_MUTATION = c->cond.mu[0];
		if(NUM_MU > 1)
			S_MUTATION = c->cond.mu[1];
	}
#endif

	if(drand() < P_MUTATION) {
		tree_mutation(&c->cond.gp, P_MUTATION);
		return true;
	}
	else {
		return false;
	}
}

_Bool cond_crossover(CL *c1, CL *c2)
{
	if(drand() < P_CROSSOVER) {
		tree_crossover(&c1->cond.gp, &c2->cond.gp);
		return true;
	}
	else {
		return false;
	}
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
	tree_print(&c->cond.gp, 0);
}  

#endif
