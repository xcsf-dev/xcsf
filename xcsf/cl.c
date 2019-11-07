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
    c->mhist = calloc(xcsf->THETA_SUB, sizeof(_Bool));
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
    if(c->exp > xcsf->THETA_DEL && c->fit / c->num < xcsf->DELTA * avg_fit) {
        return c->size * c->num * avg_fit / (c->fit / c->num);
    }
    return c->size * c->num;
}

double cl_acc(XCSF *xcsf, CL *c)
{
    if(c->err > xcsf->EPS_0) {
        return xcsf->ALPHA * pow(c->err / xcsf->EPS_0, -(xcsf->NU));
    }
    return 1;
}

void cl_update(XCSF *xcsf, CL *c, double *x, double *y, int set_num)
{
    c->exp++;
    cl_update_err(xcsf, c, y);
    cl_update_size(xcsf, c, set_num);
    cond_update(xcsf, c, x, y);
    pred_update(xcsf, c, x, y);
    act_update(xcsf, c, x, y);
}

double cl_update_err(XCSF *xcsf, CL *c, double *y)
{
    // prediction has been updated for the current input during set_pred()
    double error = (xcsf->loss_ptr)(xcsf, c->prediction, y);
    if(c->exp < 1 / xcsf->BETA) {
        c->err = (c->err * (c->exp - 1) + error) / c->exp;
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
    if(c->exp < 1 / xcsf->BETA) {
        c->size = (c->size * (c->exp - 1) + num_sum) / c->exp;
    }
    else {
        c->size += xcsf->BETA * (num_sum - c->size);
    }
    return c->size * c->num;
}

void cl_free(XCSF *xcsf, CL *c)
{
    free(c->mhist);
    free(c->prediction);
    free(c->action);
    sam_free(xcsf, c->mu);
    cond_free(xcsf, c);
    act_free(xcsf, c);
    pred_free(xcsf, c);
    free(c);
}

void cl_print(XCSF *xcsf, CL *c, _Bool printc, _Bool printa, _Bool printp)
{
    if(printc || printa || printp) {
        printf("***********************************************\n");
        if(printc) {
            printf("\nCONDITION\n");
            cond_print(xcsf, c);
        }
        if(printp) {
            printf("\nPREDICTOR\n");
            pred_print(xcsf, c);
        }
        if(printa) {
            printf("\nACTION\n");
            act_print(xcsf, c);
        }
        printf("\n");
    }
    printf("err=%f, fit=%f, num=%d, exp=%d, size=%f, time=%d\n", 
            c->err, c->fit, c->num, c->exp, c->size, c->time);
}  

_Bool cl_match(XCSF *xcsf, CL *c, double *x)
{
    _Bool m = cond_match(xcsf, c, x);
    int sub = xcsf->THETA_SUB;
    c->mhist[c->exp % sub] = m;
    return m;
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
    return false;
}

_Bool cl_general(XCSF *xcsf, CL *c1, CL *c2)
{
    if(cond_general(xcsf, c1, c2)) {
        return act_general(xcsf, c1, c2);
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
                xcsf->E_MUTATION = c->mu[2];
                if(xcsf->SAM_NUM > 3) {
                    xcsf->F_MUTATION = c->mu[3];
                }
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
    _Bool cc = cond_crossover(xcsf, c1, c2);
    _Bool pc = pred_crossover(xcsf, c1, c2);
    _Bool ac = act_crossover(xcsf, c1, c2);
    if(cc || pc || ac) {
        return true;
    }
    return false;
}  

double cl_mutation_rate(XCSF *xcsf, CL *c, int m)
{
    (void)xcsf;
    return c->mu[m];
}  

int cl_cond_size(XCSF *xcsf, CL *c)
{
    return cond_size(xcsf, c);
}

int cl_pred_size(XCSF *xcsf, CL *c)
{
    return pred_size(xcsf, c);
}

size_t cl_save(XCSF *xcsf, CL *c, FILE *fp)
{
    size_t s = 0;
    s += fwrite(c->mu, sizeof(double), xcsf->SAM_NUM, fp);
    s += fwrite(&c->err, sizeof(double), 1, fp);
    s += fwrite(&c->fit, sizeof(double), 1, fp);
    s += fwrite(&c->num, sizeof(int), 1, fp);
    s += fwrite(&c->exp, sizeof(int), 1, fp);
    s += fwrite(&c->size, sizeof(double), 1, fp);
    s += fwrite(&c->time, sizeof(int), 1, fp);
    s += fwrite(&c->m, sizeof(_Bool), 1, fp);
    s += fwrite(c->mhist, sizeof(_Bool), xcsf->THETA_SUB, fp);
    s += fwrite(c->prediction, sizeof(double), xcsf->num_y_vars, fp);
    s += fwrite(c->action, sizeof(double), xcsf->num_y_vars, fp);
    s += act_save(xcsf, c, fp);
    s += pred_save(xcsf, c, fp);
    s += cond_save(xcsf, c, fp);
    //printf("cl saved %lu elements\n", (unsigned long)s);
    return s;
}

size_t cl_load(XCSF *xcsf, CL *c, FILE *fp)
{
    size_t s = 0;
    c->mu = malloc(xcsf->SAM_NUM * sizeof(double));
    s += fread(c->mu, sizeof(double), xcsf->SAM_NUM, fp);
    s += fread(&c->err, sizeof(double), 1, fp);
    s += fread(&c->fit, sizeof(double), 1, fp);
    s += fread(&c->num, sizeof(int), 1, fp);
    s += fread(&c->exp, sizeof(int), 1, fp);
    s += fread(&c->size, sizeof(double), 1, fp);
    s += fread(&c->time, sizeof(int), 1, fp);
    s += fread(&c->m, sizeof(_Bool), 1, fp);
    c->mhist = malloc(xcsf->THETA_SUB * sizeof(_Bool));
    s += fread(c->mhist, sizeof(_Bool), xcsf->THETA_SUB, fp);
    c->prediction = malloc(xcsf->num_y_vars * sizeof(double));
    s += fread(c->prediction, sizeof(double), xcsf->num_y_vars, fp);
    c->action = malloc(xcsf->num_y_vars * sizeof(double));
    s += fread(c->action, sizeof(double), xcsf->num_y_vars, fp);
    action_set(xcsf, c);
    prediction_set(xcsf, c);
    condition_set(xcsf, c);
    s += act_load(xcsf, c, fp);
    s += pred_load(xcsf, c, fp);
    s += cond_load(xcsf, c, fp);
    //printf("cl loaded %lu elements\n", (unsigned long)s);
    return s;
}
