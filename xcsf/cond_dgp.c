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
#include "data_structures.h"
#include "cl.h"
#include "dgp.h"
#include "cond_dgp.h"

typedef struct COND_DGP {
	GRAPH dgp;
} COND_DGP;

void cond_dgp_init(XCSF *xcsf, CL *c)
{
	COND_DGP *cond = malloc(sizeof(COND_DGP));
	graph_init(xcsf, &cond->dgp, xcsf->DGP_NUM_NODES);
	c->cond = cond;
}

void cond_dgp_free(XCSF *xcsf, CL *c)
{
	COND_DGP *cond = c->cond;
	graph_free(xcsf, &cond->dgp);
	free(c->cond);
}                  

void cond_dgp_copy(XCSF *xcsf, CL *to, CL *from)
{
	COND_DGP *to_cond = to->cond;
	COND_DGP *from_cond = from->cond;
	graph_copy(xcsf, &to_cond->dgp, &from_cond->dgp);
}

void cond_dgp_rand(XCSF *xcsf, CL *c)
{
	COND_DGP *cond = c->cond;
	graph_rand(xcsf, &cond->dgp);
}

void cond_dgp_cover(XCSF *xcsf, CL *c, double *state)
{
	// generates random graphs until the network matches for input state
	do {
		cond_dgp_rand(xcsf, c);
	} while(!cond_dgp_match(xcsf, c, state));
}

_Bool cond_dgp_match(XCSF *xcsf, CL *c, double *state)
{
	// classifier matches if the first output node > 0.5
	COND_DGP *cond = c->cond;
	graph_update(xcsf, &cond->dgp, state);
	if(graph_output(xcsf, &cond->dgp, 0) > 0.5) {
		c->m = true;
	}
	else {
		c->m = false;
	}
	return c->m;
}            

_Bool cond_dgp_mutate(XCSF *xcsf, CL *c)
{
	COND_DGP *cond = c->cond;
	return graph_mutate(xcsf, &cond->dgp);
}

_Bool cond_dgp_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
	COND_DGP *cond1 = c1->cond;
	COND_DGP *cond2 = c2->cond;
	return graph_crossover(xcsf, &cond1->dgp, &cond2->dgp);
}

_Bool cond_dgp_general(XCSF *xcsf, CL *c1, CL *c2)
{
	(void)xcsf;
	(void)c1;
	(void)c2;
	return false;
}   

void cond_dgp_print(XCSF *xcsf, CL *c)
{
	(void)xcsf;
	COND_DGP *cond = c->cond;
	graph_print(xcsf, &cond->dgp);
}  
