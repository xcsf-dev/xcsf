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
	c->weights_length = (state_length*XCSF_EXPONENT)+1;
	con_init(c);
	c->weights = malloc(sizeof(double) * c->weights_length);
	for(int i = 0; i < c->weights_length; i++)
		c->weights[i] = 0.0;
#ifdef SELF_ADAPT_MUTATION
	sam_init(c);
#endif
}

void copy_cl(CL *to, CL *from)
{
	init_cl(to, from->size, from->time);
	to->weights_length = from->weights_length;
	memcpy(to->weights, from->weights, sizeof(double)*from->weights_length);
	con_copy(to, from);
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

void update_pre(CL *c, double p, double *state)
{
	double error = p - compute_pre(c, state);
	double norm = XCSF_X0 * XCSF_X0;
	for(int i = 0; i < state_length; i++)
		norm += state[i] * state[i];
	double correction = (XCSF_ETA * error) / norm;
	c->weights[0] += XCSF_X0 * correction;
	for(int i = 0; i < c->weights_length-1; i+=XCSF_EXPONENT)
		for(int j = 0; j < XCSF_EXPONENT; j++)
			c->weights[i+j+1] += correction * pow(state[i/XCSF_EXPONENT], j+1);
}

double compute_pre(CL *c, double *state)
{
	double pre = XCSF_X0 * c->weights[0];
	for(int i = 0; i < c->weights_length-1; i+=XCSF_EXPONENT)
		for(int j = 0; j < XCSF_EXPONENT; j++)
			pre += pow(state[i/XCSF_EXPONENT], j+1) * c->weights[i+j+1];
	return pre;
} 

double update_err(CL *c, double p, double *state)
{
	double pre = compute_pre(c, state);
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
	free(c->weights);
#ifdef SELF_ADAPT_MUTATION
	sam_free(c);
#endif
	free(c);
}

void print_cl(CL *c)
{
	con_print(c);
	printf("%f %f %d %d %f %d\n", c->err, c->fit, c->num, c->exp, c->size, c->time);
	printf("weights: ");
	for(int i = 0; i < c->weights_length; i++)
		printf("%f, ", c->weights[i]);
	printf("\n");
}  
