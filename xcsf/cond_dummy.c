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
 * Always matching condition module.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "cl.h"
#include "cond_dummy.h"
 
typedef struct COND_DUMMY {
	_Bool m;
	double *mu;
} COND_DUMMY;
 
void cond_dummy_init(CL *c)
{
	COND_DUMMY *cond = malloc(sizeof(COND_DUMMY));
	c->cond = cond;
	sam_init(&cond->mu);
}

void cond_dummy_free(CL *c)
{
	COND_DUMMY *cond = c->cond;
	sam_free(cond->mu);
	free(c->cond);
}
 
double cond_dummy_mu(CL *c, int m)
{
	COND_DUMMY *cond = c->cond;
	return cond->mu[m];
}
 
void cond_dummy_copy(CL *to, CL *from)
{
	(void)to;
	(void)from;
}                             

void cond_dummy_rand(CL *c)
{
	(void)c;
}

void cond_dummy_cover(CL *c, double *state)
{
	(void)c;
	(void)state;
}

_Bool cond_dummy_match(CL *c, double *state)
{
	(void)state;
	COND_DUMMY *cond = c->cond;
	cond->m = true;
	return cond->m;
}
 
_Bool cond_dummy_match_state(CL *c)
{
	COND_DUMMY *cond = c->cond;
	return cond->m;
}
 
_Bool cond_dummy_crossover(CL *c1, CL *c2) 
{
	(void)c1;
	(void)c2;
	return false;
}

_Bool cond_dummy_mutate(CL *c)
{
	(void)c;
	return false;
}

_Bool cond_dummy_subsumes(CL *c1, CL *c2)
{
	(void)c1;
	(void)c2;
	return true;
}

_Bool cond_dummy_general(CL *c1, CL *c2)
{
	(void)c1;
	(void)c2;
	return true;
}  

void cond_dummy_print(CL *c)
{
	(void)c;
}
