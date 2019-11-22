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
 */
   
/**
 * @file cond_ellipsoid.c
 * @brief Hyperellipsoid condition functions.
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
#include "cond_ellipsoid.h"

/**
 * @brief Hyperellipsoid condition data structure.
 */ 
typedef struct COND_ELLIPSOID {
    double *center; //!< centers
    double *spread; //!< spreads
} COND_ELLIPSOID;

double cond_ellipsoid_dist(XCSF *xcsf, CL *c, double *x);

void cond_ellipsoid_init(XCSF *xcsf, CL *c)
{
    COND_ELLIPSOID *new = malloc(sizeof(COND_ELLIPSOID));
    new->center = malloc(sizeof(double)*xcsf->num_x_vars);
    new->spread = malloc(sizeof(double)*xcsf->num_x_vars); 
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        new->center[i] = rand_uniform(xcsf->MIN_CON, xcsf->MAX_CON);
        new->spread[i] = rand_uniform(0.05, fabs(xcsf->MAX_CON - xcsf->MIN_CON));
    }
    c->cond = new;
}

void cond_ellipsoid_free(XCSF *xcsf, CL *c)
{
    (void)xcsf;
    COND_ELLIPSOID *cond = c->cond;
    free(cond->center);
    free(cond->spread);
    free(c->cond);
}

void cond_ellipsoid_copy(XCSF *xcsf, CL *to, CL *from)
{
    COND_ELLIPSOID *new = malloc(sizeof(COND_ELLIPSOID));
    COND_ELLIPSOID *from_cond = from->cond;
    new->center = malloc(sizeof(double)*xcsf->num_x_vars);
    new->spread = malloc(sizeof(double)*xcsf->num_x_vars); 
    memcpy(new->center, from_cond->center, sizeof(double)*xcsf->num_x_vars);
    memcpy(new->spread, from_cond->spread, sizeof(double)*xcsf->num_x_vars);
    to->cond = new;
}                             

void cond_ellipsoid_cover(XCSF *xcsf, CL *c, double *x)
{
    COND_ELLIPSOID *cond = c->cond;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        cond->center[i] = x[i];
        cond->spread[i] = rand_uniform(0.05, fabs(xcsf->MAX_CON - xcsf->MIN_CON));
    }
}

void cond_ellipsoid_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)y;
    COND_ELLIPSOID *cond = c->cond;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        cond->center[i] += xcsf->COND_ETA * (x[i] - cond->center[i]);
    }
}

_Bool cond_ellipsoid_match(XCSF *xcsf, CL *c, double *x)
{
    if(cond_ellipsoid_dist(xcsf, c, x) < 1) {
        c->m = true;
    }
    else {
        c->m = false;
    }
    return c->m;
}

double cond_ellipsoid_dist(XCSF *xcsf, CL *c, double *x)
{
    COND_ELLIPSOID *cond = c->cond;
    double dist = 0;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        double d = (x[i] - cond->center[i]) / cond->spread[i];
        dist += d*d; // squared distance
    }
    return dist;
}

_Bool cond_ellipsoid_crossover(XCSF *xcsf, CL *c1, CL *c2) 
{
    COND_ELLIPSOID *cond1 = c1->cond;
    COND_ELLIPSOID *cond2 = c2->cond;
    _Bool changed = false;
    if(rand_uniform(0,1) < xcsf->P_CROSSOVER) {
        for(int i = 0; i < xcsf->num_x_vars; i++) {
            if(rand_uniform(0,1) < 0.5) {
                double tmp = cond1->center[i];
                cond1->center[i] = cond2->center[i];
                cond2->center[i] = tmp;
                changed = true;
            }
            if(rand_uniform(0,1) < 0.5) {
                double tmp = cond1->spread[i];
                cond1->spread[i] = cond2->spread[i];
                cond2->spread[i] = tmp;
                changed = true;
            }
        }
    }
    return changed;
}

_Bool cond_ellipsoid_mutate(XCSF *xcsf, CL *c)
{
    COND_ELLIPSOID *cond = c->cond;
    _Bool changed = false;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        double orig = cond->center[i];
        cond->center[i] += rand_normal(0, xcsf->S_MUTATION);
        if(cond->center[i] < xcsf->MIN_CON) {
            cond->center[i] = xcsf->MIN_CON;
        }
        else if(cond->center[i] > xcsf->MAX_CON) {
            cond->center[i] = xcsf->MAX_CON;
        }
        if(orig != cond->center[i]) {
            changed = true;
        }
    }
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        double orig = cond->spread[i];
        cond->spread[i] += rand_normal(0, xcsf->S_MUTATION);
        if(orig != cond->spread[i]) {
            changed = true;
        }
    }
    return changed;   
}

_Bool cond_ellipsoid_general(XCSF *xcsf, CL *c1, CL *c2)
{
    // returns whether cond1 is more general than cond2
    COND_ELLIPSOID *cond1 = c1->cond;
    COND_ELLIPSOID *cond2 = c2->cond;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        double l1 = cond1->center[i] - cond1->spread[i];
        double l2 = cond2->center[i] - cond2->spread[i];
        double u1 = cond1->center[i] + cond1->spread[i];
        double u2 = cond2->center[i] + cond2->spread[i];
        if(l1 > l2 || u1 < u2) {
            return false;
        }
    }
    return true;
}  

void cond_ellipsoid_print(XCSF *xcsf, CL *c)
{
    COND_ELLIPSOID *cond = c->cond;
    printf("ellipsoid:");
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        printf(" (%5f, ", cond->center[i]);
        printf("%5f)", cond->spread[i]);
    }
    printf("\n");
}

int cond_ellipsoid_size(XCSF *xcsf, CL *c)
{
    (void)c;
    return xcsf->num_x_vars * 2;
}

size_t cond_ellipsoid_save(XCSF *xcsf, CL *c, FILE *fp)
{
    size_t s = 0;
    COND_ELLIPSOID *cond = c->cond;
    s += fwrite(cond->center, sizeof(double), xcsf->num_x_vars, fp);
    s += fwrite(cond->spread, sizeof(double), xcsf->num_x_vars, fp);
    //printf("ellipsoid saved %lu elements\n", (unsigned long)s);
    return s;
}

size_t cond_ellipsoid_load(XCSF *xcsf, CL *c, FILE *fp)
{
    size_t s = 0;
    COND_ELLIPSOID *new = malloc(sizeof(COND_ELLIPSOID));
    new->center = malloc(sizeof(double)*xcsf->num_x_vars);
    new->spread = malloc(sizeof(double)*xcsf->num_x_vars);
    s += fread(new->center, sizeof(double), xcsf->num_x_vars, fp);
    s += fread(new->spread, sizeof(double), xcsf->num_x_vars, fp);
    c->cond = new;
    //printf("ellipsoid loaded %lu elements\n", (unsigned long)s);
    return s;
}
