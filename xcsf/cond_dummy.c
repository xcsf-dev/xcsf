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
#include "xcsf.h"
#include "util.h"
#include "cl.h"
#include "condition.h"
#include "cond_dummy.h"

void cond_dummy_init(XCSF *xcsf, CL *c)
{
	(void)xcsf; (void)c;
}

void cond_dummy_free(XCSF *xcsf, CL *c)
{
	(void)xcsf; (void)c;
}

void cond_dummy_copy(XCSF *xcsf, CL *to, CL *from)
{
	(void)xcsf; (void)to; (void)from;
}                             

void cond_dummy_rand(XCSF *xcsf, CL *c)
{
	(void)xcsf; (void)c;
}

void cond_dummy_cover(XCSF *xcsf, CL *c, double *state)
{
	(void)xcsf; (void)c; (void)state;
}

_Bool cond_dummy_match(XCSF *xcsf, CL *c, double *state)
{
	(void)xcsf; (void)state;
	c->m = true;
	return c->m;
}

_Bool cond_dummy_crossover(XCSF *xcsf, CL *c1, CL *c2) 
{
	(void)xcsf; (void)c1; (void)c2;
	return false;
}

_Bool cond_dummy_mutate(XCSF *xcsf, CL *c)
{
	(void)xcsf; (void)c;
	return false;
}

_Bool cond_dummy_general(XCSF *xcsf, CL *c1, CL *c2)
{
	(void)xcsf; (void)c1; (void)c2;
	return true;
}  

void cond_dummy_print(XCSF *xcsf, CL *c)
{
	(void)xcsf; (void)c;
}
