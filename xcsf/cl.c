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
#include "data_structures.h"
#include "cl.h"
#include "cond_dummy.h"
#include "cond_rectangle.h"
#include "cond_ellipsoid.h"
#include "cond_gp.h"
#include "cond_dgp.h"
#include "cond_neural.h"
#include "pred_nlms.h"
#include "pred_rls.h"
#include "pred_neural.h"
#include "rule_dgp.h"
#include "rule_neural.h"
#include "loss.h"
#include "sam.h"

double cl_update_err(XCSF *xcsf, CL *c, double *y);
double cl_update_size(XCSF *xcsf, CL *c, double num_sum);

void cl_init(XCSF *xcsf, CL *c, int size, int time)
{
    c->fit = xcsf->INIT_FITNESS;
    c->err = xcsf->INIT_ERROR;
    c->num = 1;
    c->exp = 0;
    c->size = size;
    c->time = time;
    c->prediction = calloc(xcsf->num_y_vars, sizeof(double));
    c->m = false;

    switch(xcsf->PRED_TYPE) {
        case 0:
        case 1:
            c->pred_vptr = &pred_nlms_vtbl;
            break;
        case 2:
        case 3:
            c->pred_vptr = &pred_rls_vtbl;
            break;
        case 4:
            c->pred_vptr = &pred_neural_vtbl;
            break;
        default:
            printf("Invalid prediction type specified: %d\n", xcsf->PRED_TYPE);
            exit(EXIT_FAILURE);
    }

    switch(xcsf->COND_TYPE) {
        case -1:
            c->cond_vptr = &cond_dummy_vtbl;
            break;
        case 0:
            c->cond_vptr = &cond_rectangle_vtbl;
            break;
        case 1:
            c->cond_vptr = &cond_ellipsoid_vtbl;
            break;
        case 2:
            c->cond_vptr = &cond_neural_vtbl;
            break;
        case 3:
            c->cond_vptr = &cond_gp_vtbl;
            break;
        case 4:
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
            printf("Invalid condition type specified: %d\n", xcsf->COND_TYPE);
            exit(EXIT_FAILURE);
    }

    sam_init(xcsf, &c->mu);
    cond_init(xcsf, c);
    pred_init(xcsf, c);
}

void cl_copy(XCSF *xcsf, CL *to, CL *from)
{
    cl_init(xcsf, to, from->size, from->time);
	sam_copy(xcsf, to->mu, from->mu);
    cond_copy(xcsf, to, from);
    pred_copy(xcsf, to, from);
}

_Bool cl_subsumer(XCSF *xcsf, CL *c)
{
    if(c->exp > xcsf->THETA_SUB && c->err < xcsf->EPS_0) {
        return true;
    }
    else {
        return false;
    }
}

double cl_del_vote(XCSF *xcsf, CL *c, double avg_fit)
{
    if(c->fit / c->num >= xcsf->DELTA * avg_fit || c->exp < xcsf->THETA_DEL) {
        return c->size * c->num;
    }
    return c->size * c->num * avg_fit / (c->fit / c->num); 
}

double cl_acc(XCSF *xcsf, CL *c)
{
    if(c->err <= xcsf->EPS_0) {
        return 1.0;
    }
    else {
        return xcsf->ALPHA * pow(c->err / xcsf->EPS_0, -(xcsf->NU));
    }
}

void cl_update(XCSF *xcsf, CL *c, double *x, double *y, int set_num)
{
    c->exp++;
    cl_update_err(xcsf, c, y);
    pred_update(xcsf, c, x, y);
    cl_update_size(xcsf, c, set_num);
}

double cl_update_err(XCSF *xcsf, CL *c, double *y)
{
    // prediction has been updated for the current input during set_pred()
    double error = (xcsf->loss_ptr)(xcsf, c->prediction, y);

    if(c->exp < 1.0/xcsf->BETA) {
        c->err = (c->err * (c->exp-1.0) + error) / (double)c->exp;
    }
    else {
        c->err += xcsf->BETA * (error - c->err);
    }
    return c->err * c->num;
}

void cl_update_fit(XCSF *xcsf, CL *c, double acc_sum, double acc)
{
    c->fit += xcsf->BETA * ((acc * c->num) / acc_sum - c->fit);
}

double cl_update_size(XCSF *xcsf, CL *c, double num_sum)
{
    if(c->exp < 1.0/xcsf->BETA) {
        c->size = (c->size * (c->exp-1.0) + num_sum) / (double)c->exp; 
    }
    else {
        c->size += xcsf->BETA * (num_sum - c->size);
    }
    return c->size * c->num;
}

void cl_free(XCSF *xcsf, CL *c)
{
    free(c->prediction);
    sam_free(xcsf, c->mu);
    cond_free(xcsf, c);
    pred_free(xcsf, c);
    free(c);
}

void cl_print(XCSF *xcsf, CL *c, _Bool print_cond, _Bool print_pred)
{
    if(print_cond || print_pred) {
        printf("***********************************************\n");
    }
    if(print_cond) {
        cond_print(xcsf, c);
    }
    if(print_pred) {
        pred_print(xcsf, c);
    }
    printf("err=%f, fit=%f, num=%d, exp=%d, size=%f, time=%d\n", 
            c->err, c->fit, c->num, c->exp, c->size, c->time);
}  

void cl_cover(XCSF *xcsf, CL *c, double *x)
{
    cond_cover(xcsf, c, x);
}

_Bool cl_general(XCSF *xcsf, CL *c1, CL *c2)
{
    return cond_general(xcsf, c1, c2);
}

void cl_rand(XCSF *xcsf, CL *c)
{
    cond_rand(xcsf, c);
}

_Bool cl_match(XCSF *xcsf, CL *c, double *x)
{
    return cond_match(xcsf, c, x);
}

_Bool cl_m(XCSF *xcsf, CL *c)
{
    (void)xcsf;
    return c->m;
}

double *cl_predict(XCSF *xcsf, CL *c, double *x)
{
    return pred_compute(xcsf, c, x);
}

_Bool cl_mutate(XCSF *xcsf, CL *c)
{
    // apply mutation rates
	if(xcsf->SAM_NUM > 0) {
		xcsf->P_MUTATION = c->mu[0];
		if(xcsf->SAM_NUM > 1) {
			xcsf->S_MUTATION = c->mu[1];
			if(xcsf->SAM_NUM > 2) {
				xcsf->P_FUNC_MUTATION = c->mu[2];
			}
		}
	} 
    // mutate condition
    return cond_mutate(xcsf, c);
}

_Bool cl_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    return cond_crossover(xcsf, c1, c2);
}  

double cl_mutation_rate(XCSF *xcsf, CL *c, int m)
{
    (void)xcsf;
    return c->mu[m];
}
