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
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "loss.h"
#include "sam.h"
#include "condition.h"
#include "prediction.h"
#include "action.h"
#include "cl.h"

double cl_update_err(XCSF *xcsf, CL *c, double *y);
double cl_update_size(XCSF *xcsf, CL *c, double num_sum);
_Bool cl_cond_type_identical(CL *c1, CL *c2);
_Bool cl_pred_type_identical(CL *c1, CL *c2);
_Bool cl_act_type_identical(CL *c1, CL *c2);

void cl_init(XCSF *xcsf, CL *c, int size, int time)
{
    c->fit = xcsf->INIT_FITNESS;
    c->err = xcsf->INIT_ERROR;
    c->num = 1;
    c->exp = 0;
    c->size = size;
    c->time = time;
    c->prediction = calloc(xcsf->num_y_vars, sizeof(double));
    c->action = calloc(xcsf->num_y_vars, sizeof(double));
    c->m = false;
    sam_init(xcsf, &c->mu);
}

void cl_copy(XCSF *xcsf, CL *to, CL *from)
{
    // copy functions
    to->cond_vptr = from->cond_vptr;
    to->pred_vptr = from->pred_vptr;
    to->act_vptr = from->act_vptr;
    // copy structures
    sam_copy(xcsf, to->mu, from->mu);
    act_copy(xcsf, to, from);
    cond_copy(xcsf, to, from);
    pred_copy(xcsf, to, from);
}

void cl_cover(XCSF *xcsf, CL *c, double *x)
{
    cl_rand(xcsf, c);
    cond_cover(xcsf, c, x);
}

void cl_rand(XCSF *xcsf, CL *c)
{
    // set functions
    action_set(xcsf, c);
    prediction_set(xcsf, c);
    condition_set(xcsf, c); 
    // initialise structures
    cond_init(xcsf, c);
    pred_init(xcsf, c);
    act_init(xcsf, c);
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
    cond_update(xcsf, c, x, y);
    pred_update(xcsf, c, x, y);
    act_update(xcsf, c, x, y);
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
    free(c->action);
    sam_free(xcsf, c->mu);
    cond_free(xcsf, c);
    act_free(xcsf, c);
    pred_free(xcsf, c);
    free(c);
}

void cl_print(XCSF *xcsf, CL *c, _Bool print_cond, _Bool print_pred)
{
    if(print_cond || print_pred) {
        printf("***********************************************\n");
    }
    if(print_cond) {
        printf("\nCONDITION\n");
        cond_print(xcsf, c);
    }
    if(print_pred) {
        printf("\nPREDICTOR\n");
        pred_print(xcsf, c);
        printf("\nACTION\n");
        act_print(xcsf, c);
        printf("\n");
    }
    printf("err=%f, fit=%f, num=%d, exp=%d, size=%f, time=%d\n", 
            c->err, c->fit, c->num, c->exp, c->size, c->time);
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

_Bool cl_subsumer(XCSF *xcsf, CL *c)
{
    if(c->exp > xcsf->THETA_SUB && c->err < xcsf->EPS_0) {
        return true;
    }
    else {
        return false;
    }
}

_Bool cl_general(XCSF *xcsf, CL *c1, CL *c2)
{
    if(cl_cond_type_identical(c1, c2) && cond_general(xcsf, c1, c2)) {
        if(cl_act_type_identical(c1, c2)) {
            return act_general(xcsf, c1, c2);
        }
    }
    return false;
} 

_Bool cl_mutate(XCSF *xcsf, CL *c)
{
    if(xcsf->SAM_NUM > 0) {
        xcsf->S_MUTATION = c->mu[0];
        if(xcsf->SAM_NUM > 1) {
            xcsf->P_MUTATION = c->mu[1];
            if(xcsf->SAM_NUM > 2) {
                xcsf->P_FUNC_MUTATION = c->mu[2];
            }
        }
    } 
    _Bool cm = cond_mutate(xcsf, c);
    _Bool pm = pred_mutate(xcsf, c);
    _Bool am = act_mutate(xcsf, c);
    if(cm || pm || am) {
        return true;
    }
    return false;
}

_Bool cl_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    _Bool cc = false, pc = false, ac = false;
    if(cl_cond_type_identical(c1, c2)) {
        cc = cond_crossover(xcsf, c1, c2);
    }
    if(cl_pred_type_identical(c1, c2)) {
        pc = pred_crossover(xcsf, c1, c2);
    }
    if(cl_act_type_identical(c1, c2)) {
        ac = act_crossover(xcsf, c1, c2);
    }
    if(cc || pc || ac) {
        return true;
    }
    return false;
}  

_Bool cl_cond_type_identical(CL *c1, CL *c2)
{
    if(c1->cond_vptr == c2->cond_vptr) {
        return true;
    }
    return false;
}

_Bool cl_pred_type_identical(CL *c1, CL *c2)
{
    if(c1->pred_vptr == c2->pred_vptr) {
        return true;
    }
    return false;
}

_Bool cl_act_type_identical(CL *c1, CL *c2)
{
    if(c1->act_vptr == c2->act_vptr) {
        return true;
    }
    return false;
}

double cl_mutation_rate(XCSF *xcsf, CL *c, int m)
{
    (void)xcsf;
    return c->mu[m];
}  
