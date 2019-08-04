/*
 * Copyright (C) 2015--2019 Richard Preen <rpreen@gmail.com>
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
 * The classifier module.  
 *
 * Performs general operations applied to an individual classifier: creation,
 * copying, deletion, updating, and printing.
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
#include "cond_rect.h"
#include "cond_gp.h"
#include "cond_dgp.h"
#include "cond_neural.h"
#include "pred_nlms.h"
#include "pred_rls.h"
#include "pred_neural.h"
#include "rule_dgp.h"
#include "rule_neural.h"

double cl_update_err(CL *c, double *y);
double cl_update_size(CL *c, double num_sum);

void cl_init(CL *c, int size, int time)
{
	c->fit = INIT_FITNESS;
	c->err = INIT_ERROR;
	c->num = 1;
	c->exp = 0;
	c->size = size;
	c->time = time;

	switch(PRED_TYPE) {
		case 0:
			c->pred_vptr = &pred_nlms_vtbl;
			break;
		case 1:
			c->pred_vptr = &pred_neural_vtbl;
			break;
		case 2:
			c->pred_vptr = &pred_rls_vtbl;
			break;
		default:
			printf("Invalid prediction type specified: %d\n", PRED_TYPE);
			exit(EXIT_FAILURE);
	}

	switch(COND_TYPE) {
		case -1:
			c->cond_vptr = &cond_dummy_vtbl;
			break;
		case 0:
			c->cond_vptr = &cond_rect_vtbl;
			break;
		case 1:
			c->cond_vptr = &cond_neural_vtbl;
			break;
		case 2:
			c->cond_vptr = &cond_gp_vtbl;
			break;
		case 3:
			c->cond_vptr = &cond_dgp_vtbl;
			break;
		case 11:
			c->cond_vptr = &rule_dgp_cond_vtbl;
			c->pred_vptr = &rule_dgp_pred_vtbl;
			break;
		case 12:
			c->cond_vptr = &rule_neural_cond_vtbl;
			c->pred_vptr = &rule_neural_pred_vtbl;
			break;
		default:
			printf("Invalid condition type specified: %d\n", COND_TYPE);
			exit(EXIT_FAILURE);
	}

	cond_init(c);
	pred_init(c);
}

void cl_copy(CL *to, CL *from)
{
	cl_init(to, from->size, from->time);
	cond_copy(to, from);
	pred_copy(to, from);
}

_Bool cl_subsumer(CL *c)
{
	if(c->exp > THETA_SUB && c->err < EPS_0)
		return true;
	else
		return false;
}

double cl_del_vote(CL *c, double avg_fit)
{
	if(c->fit / c->num >= DELTA * avg_fit || c->exp < THETA_DEL)
		return c->size * c->num;
	return c->size * c->num * avg_fit / (c->fit / c->num); 
}

double cl_acc(CL *c)
{
	if(c->err <= EPS_0)
		return 1.0;
	else
		return ALPHA * pow(c->err / EPS_0, -NU);
}

void cl_update(CL *c, double *x, double *y, int set_num)
{
	c->exp++;
	cl_update_err(c, y);
	pred_update(c, y, x);
	cl_update_size(c, set_num);
}

double cl_update_err(CL *c, double *y)
{
	// calculate MSE
	double error = 0.0;
	for(int i = 0; i < num_y_vars; i++) {
		double pre = pred_pre(c, i);
		error += (y[i] - pre) * (y[i] - pre);
	}
	error /= (double)num_y_vars;

	// prediction has been updated for the current state during set_pred()
	if(c->exp < 1.0/BETA) {
		c->err = (c->err * (c->exp-1.0) + error) / (double)c->exp;
	}
	else {
		c->err += BETA * (error - c->err);
	}
	return c->err * c->num;
}
 
void cl_update_fit(CL *c, double acc_sum, double acc)
{
	c->fit += BETA * ((acc * c->num) / acc_sum - c->fit);
}

double cl_update_size(CL *c, double num_sum)
{
	if(c->exp < 1.0/BETA)
		c->size = (c->size * (c->exp-1.0) + num_sum) / (double)c->exp; 
	else
		c->size += BETA * (num_sum - c->size);
	return c->size * c->num;
}

void cl_free(CL *c)
{
	cond_free(c);
	pred_free(c);
	free(c);
}

void cl_print(CL *c)
{
	cond_print(c);
	pred_print(c);
	printf("%f %f %d %d %f %d\n", c->err, c->fit, c->num, c->exp, c->size, c->time);
}  

void cl_cover(CL *c, double *x)
{
	cond_cover(c, x);
}

_Bool cl_general(CL *c1, CL *c2)
{
    return cond_general(c1, c2);
}

_Bool cl_subsumes(CL *c1, CL *c2)
{
	return cond_subsumes(c1, c2);
}

void cl_rand(CL *c)
{
	cond_rand(c);
}

_Bool cl_match(CL *c, double *x)
{
	return cond_match(c, x);
}

_Bool cl_match_state(CL *c)
{
	return cond_match_state(c);
}

double *cl_predict(CL *c, double *x)
{
	return pred_compute(c, x);
}

_Bool cl_mutate(CL *c)
{
	return cond_mutate(c);
}
 
_Bool cl_crossover(CL *c1, CL *c2)
{
	return cond_crossover(c1, c2);
}  

double cl_mutation_rate(CL *c, int m)
{
	return cond_mu(c, m);
}
