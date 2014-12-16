/*
 * Copyright (C) 2015 Richard Preen <rpreen@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "cl.h"

void init_cl(CL *c, int size, int time)
{
	c->fit = INIT_FITNESS;
	c->err = INIT_ERROR;
	c->num = 1;
	c->exp = 0;
	c->size = size;
	c->time = time;
	con_init(c);
	pred_init(c);
#ifdef SELF_ADAPT_MUTATION
	sam_init(c);
#endif
}

void copy_cl(CL *to, CL *from)
{
	init_cl(to, from->size, from->time);
	con_copy(to, from);
	pred_copy(to, from);
#ifdef SELF_ADAPT_MUTATION
	sam_copy(to, from);
#endif
}

_Bool subsumer(CL *c)
{
	if(c->exp > THETA_SUB && c->err < EPS_0)
		return true;
	else
		return false;
}

double del_vote(CL *c, double avg_fit)
{
	if(c->fit / c->num >= DELTA * avg_fit || c->exp < THETA_DEL)
		return c->size * c->num;
	return c->size * c->num * avg_fit / (c->fit / c->num); 
}

double update_err(CL *c, double p, double *state)
{
	double pre = pred_compute(c, state);
	if(c->exp < 1.0/BETA) 
		c->err = (c->err * (c->exp-1.0) + fabs(p - pre)) / (double)c->exp;
	else
		c->err += BETA * (fabs(p - pre) - c->err);
	return c->err * c->num;
}

double acc(CL *c)
{
	if(c->err <= EPS_0)
		return 1.0;
	else
		return ALPHA * pow(c->err / EPS_0, -NU);
}

void update_fit(CL *c, double acc_sum, double acc)
{
	c->fit += BETA * ((acc * c->num) / acc_sum - c->fit);
}

double update_size(CL *c, double num_sum)
{
	if(c->exp < 1.0/BETA)
		c->size = (c->size * (c->exp-1.0) + num_sum) / (double)c->exp; 
	else
		c->size += BETA * (num_sum - c->size);
	return c->size * c->num;
}

void free_cl(CL *c)
{
	con_free(c);
	pred_free(c);
#ifdef SELF_ADAPT_MUTATION
	sam_free(c);
#endif
	free(c);
}

void print_cl(CL *c)
{
	con_print(c);
	pred_print(c);
	printf("%f %f %d %d %f %d\n", c->err, c->fit, c->num, c->exp, c->size, c->time);
}  
