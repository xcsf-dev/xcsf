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

#if CON == -1

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
	(void)c;
#ifdef SAM
	sam_init(&c->cond.mu);
#endif
}

void cond_free(CL *c)
{
	(void)c;
#ifdef SAM
	sam_free(c->cond.mu);
#endif
}

void cond_copy(CL *to, CL *from)
{
	(void)to;
	(void)from;
}                             

void cond_rand(CL *c)
{
	(void)c;
}

void cond_cover(CL *c, double *state)
{
	(void)c;
	(void)state;
}

_Bool cond_match(CL *c, double *state)
{
	(void)state;
	c->cond.m = true;
	return c->cond.m;
}

_Bool cond_crossover(CL *c1, CL *c2) 
{
	(void)c1;
	(void)c2;
	return false;
}

_Bool cond_mutate(CL *c)
{
	(void)c;
	return false;
}

_Bool cond_subsumes(CL *c1, CL *c2)
{
	(void)c1;
	(void)c2;
	return true;
}

_Bool cond_general(CL *c1, CL *c2)
{
	(void)c1;
	(void)c2;
	return true;
}  

void cond_print(CL *c)
{
	(void)c;
}
#endif
