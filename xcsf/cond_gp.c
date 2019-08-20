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
#include "xcsf.h"
#include "random.h"
#include "cl.h"
#include "condition.h"
#include "cond_gp.h"
#include "gp.h"

typedef struct COND_GP {
	GP_TREE gp;
} COND_GP;

void cond_gp_init(XCSF *xcsf, CL *c)
{
	COND_GP *cond = malloc(sizeof(COND_GP));
	tree_init(xcsf, &cond->gp);
	c->cond = cond;
}

void cond_gp_free(XCSF *xcsf, CL *c)
{
	COND_GP *cond = c->cond;
	tree_free(xcsf, &cond->gp);
	free(c->cond);
}

void cond_gp_copy(XCSF *xcsf, CL *to, CL *from)
{
	COND_GP *to_cond = to->cond;
	COND_GP *from_cond = from->cond;
	tree_copy(xcsf, &to_cond->gp, &from_cond->gp);
}

void cond_gp_rand(XCSF *xcsf, CL *c)
{
	COND_GP *cond = c->cond;
	tree_free(xcsf, &cond->gp);
	tree_rand(xcsf, &cond->gp);
}

void cond_gp_cover(XCSF *xcsf, CL *c, double *state)
{
	// generates random weights until the tree matches for input state
	do {
		cond_gp_rand(xcsf, c);
	} while(!cond_gp_match(xcsf, c, state));
}

_Bool cond_gp_match(XCSF *xcsf, CL *c, double *state)
{
	// classifier matches if the tree output > 0.5
	COND_GP *cond = c->cond;
	cond->gp.p = 0;
	double result = tree_eval(xcsf, &cond->gp, state);
	if(result > 0.5) {
		c->m = true;
	}
	else {
		c->m = false;
	}
	return c->m;
}    

_Bool cond_gp_mutate(XCSF *xcsf, CL *c)
{
	COND_GP *cond = c->cond;
	if(rand_uniform(0,1) < xcsf->P_MUTATION) {
		tree_mutation(xcsf, &cond->gp, xcsf->P_MUTATION);
		return true;
	}
	else {
		return false;
	}
}

_Bool cond_gp_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
	COND_GP *cond1 = c1->cond;
	COND_GP *cond2 = c2->cond;
	if(rand_uniform(0,1) < xcsf->P_CROSSOVER) {
		tree_crossover(xcsf, &cond1->gp, &cond2->gp);
		return true;
	}
	else {
		return false;
	}
}

_Bool cond_gp_subsumes(XCSF *xcsf, CL *c1, CL *c2)
{
	(void)xcsf; (void)c1; (void)c2;
	return false;
}

_Bool cond_gp_general(XCSF *xcsf, CL *c1, CL *c2)
{
	(void)xcsf; (void)c1; (void)c2;
	return false;
}   

void cond_gp_print(XCSF *xcsf, CL *c)
{
    COND_GP *cond = c->cond;
    printf("GP tree: ");
    tree_print(xcsf, &cond->gp, 0);
    printf("\n");
}  
