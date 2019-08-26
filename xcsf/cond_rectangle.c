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
 * The hyperrectangle classifier condition module.
 *
 * Provides functionality to create real-valued hyperrectangle conditions
 * whereby a classifier matches for a given problem instance if, and
 * only if, all of the current state variables fall within the area covered.
 * Includes operations for copying, mutating, printing, etc.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "condition.h"
#include "cond_rectangle.h"

typedef struct COND_RECTANGLE {
    double *lower;
    double *upper;
} COND_RECTANGLE;

void cond_rectangle_order(double *l, double *u);

void cond_rectangle_init(XCSF *xcsf, CL *c)
{
    COND_RECTANGLE *new = malloc(sizeof(COND_RECTANGLE));
    new->lower = malloc(sizeof(double)*xcsf->num_x_vars);
    new->upper = malloc(sizeof(double)*xcsf->num_x_vars); 
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        new->lower[i] = rand_uniform(xcsf->MIN_CON, xcsf->MAX_CON);
        new->upper[i] = rand_uniform(xcsf->MIN_CON, xcsf->MAX_CON);
        cond_rectangle_order(&new->lower[i], &new->upper[i]);
    }  
    c->cond = new;     
}

void cond_rectangle_free(XCSF *xcsf, CL *c)
{
    (void)xcsf;
    COND_RECTANGLE *cond = c->cond;
    free(cond->lower);
    free(cond->upper);
    free(c->cond);
}

void cond_rectangle_copy(XCSF *xcsf, CL *to, CL *from)
{
    COND_RECTANGLE *new = malloc(sizeof(COND_RECTANGLE));
    COND_RECTANGLE *from_cond = from->cond;
    new->lower = malloc(sizeof(double)*xcsf->num_x_vars);
    new->upper = malloc(sizeof(double)*xcsf->num_x_vars); 
    memcpy(new->lower, from_cond->lower, sizeof(double)*xcsf->num_x_vars);
    memcpy(new->upper, from_cond->upper, sizeof(double)*xcsf->num_x_vars);
    to->cond = new;
}                             

void cond_rectangle_order(double *l, double *u)
{
    if(*l > *u) {
        double tmp = *l;
        *l = *u;
        *u = tmp;
    }
}

void cond_rectangle_cover(XCSF *xcsf, CL *c, double *x)
{
    COND_RECTANGLE *cond = c->cond;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        cond->lower[i] = x[i] - rand_uniform(xcsf->MIN_CON, xcsf->MAX_CON) * 0.5;
        cond->upper[i] = x[i] + rand_uniform(xcsf->MIN_CON, xcsf->MAX_CON) * 0.5;
    }
}
 
void cond_rectangle_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)y;
    COND_RECTANGLE *cond = c->cond;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        double center = ((cond->upper[i] - cond->lower[i]) * 0.5) + cond->lower[i];
        double delta = xcsf->BETA * (x[i] - center);
        cond->lower[i] += delta;
        cond->upper[i] += delta;
    }
}
 
_Bool cond_rectangle_match(XCSF *xcsf, CL *c, double *x)
{
    COND_RECTANGLE *cond = c->cond;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        if(cond->lower[i] > x[i] || cond->upper[i] < x[i]) {
            c->m = false;
            return c->m;
        }
    }
    c->m = true;
    return c->m;
}

_Bool cond_rectangle_crossover(XCSF *xcsf, CL *c1, CL *c2) 
{
    COND_RECTANGLE *cond1 = c1->cond;
    COND_RECTANGLE *cond2 = c2->cond;
    _Bool changed = false;
    if(rand_uniform(0,1) < xcsf->P_CROSSOVER) {
        for(int i = 0; i < xcsf->num_x_vars; i++) {
            if(rand_uniform(0,1) < 0.5) {
                double tmp = cond1->lower[i];
                cond1->lower[i] = cond2->lower[i];
                cond2->lower[i] = tmp;
                changed = true;
            }
            if(rand_uniform(0,1) < 0.5) {
                double tmp = cond1->upper[i];
                cond1->upper[i] = cond2->upper[i];
                cond2->upper[i] = tmp;
                changed = true;
            }
            cond_rectangle_order(&cond1->lower[i], &cond1->upper[i]);
            cond_rectangle_order(&cond2->lower[i], &cond2->upper[i]);
        }
    }
    return changed;
}

_Bool cond_rectangle_mutate(XCSF *xcsf, CL *c)
{
    COND_RECTANGLE *cond = c->cond;
    _Bool changed = false;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        if(rand_uniform(0,1) < xcsf->P_MUTATION) {
            cond->lower[i] += rand_uniform(-1,1) * xcsf->S_MUTATION;
            changed = true;
        }
        if(rand_uniform(0,1) < xcsf->P_MUTATION) {
            cond->upper[i] += rand_uniform(-1,1) * xcsf->S_MUTATION;
            changed = true;
        }
        cond_rectangle_order(&cond->lower[i], &cond->upper[i]);
    }
    return changed;
}

_Bool cond_rectangle_general(XCSF *xcsf, CL *c1, CL *c2)
{
    // returns whether cond1 is more general than cond2
    COND_RECTANGLE *cond1 = c1->cond;
    COND_RECTANGLE *cond2 = c2->cond;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        if(cond1->lower[i] > cond2->lower[i] 
                || cond1->upper[i] < cond2->upper[i]) {
            return false;
        }
    }
    return true;
}  

void cond_rectangle_print(XCSF *xcsf, CL *c)
{
    COND_RECTANGLE *cond = c->cond;
    printf("rectangle:");
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        printf(" (%5f, ", cond->lower[i]);
        printf("%5f)", cond->upper[i]);
    }
    printf("\n");
}
