/*
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
 */
 
/**
 * @file cond_rectangle.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Hyperrectangle condition functions.
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

/**
 * @brief Hyperrectangle condition data structure.
 */ 
typedef struct COND_RECTANGLE {
    double *center; //!< Centers
    double *spread; //!< Spreads
} COND_RECTANGLE;

static double cond_rectangle_dist(const XCSF *xcsf, const CL *c, const double *x);

void cond_rectangle_init(const XCSF *xcsf, CL *c)
{
    COND_RECTANGLE *new = malloc(sizeof(COND_RECTANGLE));
    new->center = malloc(sizeof(double) * xcsf->num_x_vars);
    new->spread = malloc(sizeof(double) * xcsf->num_x_vars);
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        new->center[i] = rand_uniform(xcsf->COND_MIN, xcsf->COND_MAX);
        new->spread[i] = rand_uniform(xcsf->COND_SMIN, fabs(xcsf->COND_MAX - xcsf->COND_MIN));
    }  
    c->cond = new;     
}

void cond_rectangle_free(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const COND_RECTANGLE *cond = c->cond;
    free(cond->center);
    free(cond->spread);
    free(c->cond);
}

void cond_rectangle_copy(const XCSF *xcsf, CL *to, const CL *from)
{
    COND_RECTANGLE *new = malloc(sizeof(COND_RECTANGLE));
    const COND_RECTANGLE *from_cond = from->cond;
    new->center = malloc(sizeof(double) * xcsf->num_x_vars);
    new->spread = malloc(sizeof(double) * xcsf->num_x_vars);
    memcpy(new->center, from_cond->center, sizeof(double) * xcsf->num_x_vars);
    memcpy(new->spread, from_cond->spread, sizeof(double) * xcsf->num_x_vars);
    to->cond = new;
}                             

void cond_rectangle_cover(const XCSF *xcsf, CL *c, const double *x)
{
    const COND_RECTANGLE *cond = c->cond;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        cond->center[i] = x[i];
        cond->spread[i] = rand_uniform(xcsf->COND_SMIN, fabs(xcsf->COND_MAX - xcsf->COND_MIN));
    }
}

void cond_rectangle_update(const XCSF *xcsf, CL *c, const double *x, const double *y)
{
    (void)y;
    if(xcsf->COND_ETA > 0) {
        const COND_RECTANGLE *cond = c->cond;
        for(int i = 0; i < xcsf->num_x_vars; i++) {
            cond->center[i] += xcsf->COND_ETA * (x[i] - cond->center[i]);
        }
    }
}

_Bool cond_rectangle_match(const XCSF *xcsf, CL *c, const double *x)
{
    if(cond_rectangle_dist(xcsf, c, x) < 1) {
        c->m = true;
    }
    else {
        c->m = false;
    }
    return c->m;
}

static double cond_rectangle_dist(const XCSF *xcsf, const CL *c, const double *x)
{
    const COND_RECTANGLE *cond = c->cond;
    double dist = 0;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        double d = fabs((x[i] - cond->center[i]) / cond->spread[i]);
        if(d > dist) {
            dist = d; // max distance
        }
    }
    return dist;
}

_Bool cond_rectangle_crossover(const XCSF *xcsf, CL *c1, CL *c2) 
{
    const COND_RECTANGLE *cond1 = c1->cond;
    const COND_RECTANGLE *cond2 = c2->cond;
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

_Bool cond_rectangle_mutate(const XCSF *xcsf, const CL *c)
{
    const COND_RECTANGLE *cond = c->cond;
    _Bool changed = false;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        // centers
        double orig = cond->center[i];
        cond->center[i] += rand_normal(0, xcsf->S_MUTATION);
        cond->center[i] = constrain(xcsf->COND_MIN, xcsf->COND_MAX, cond->center[i]);
        if(orig != cond->center[i]) {
            changed = true;
        }
        // spreads
        orig = cond->spread[i];
        cond->spread[i] += rand_normal(0, xcsf->S_MUTATION);
        if(orig != cond->spread[i]) {
            changed = true;
        }
    }
    return changed;
}

_Bool cond_rectangle_general(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    // returns whether cond1 is more general than cond2
    const COND_RECTANGLE *cond1 = c1->cond;
    const COND_RECTANGLE *cond2 = c2->cond;
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

void cond_rectangle_print(const XCSF *xcsf, const CL *c)
{
    const COND_RECTANGLE *cond = c->cond;
    printf("rectangle:");
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        printf(" (c=%5f, ", cond->center[i]);
        printf("s=%5f)", cond->spread[i]);
    }
    printf("\n");
}

int cond_rectangle_size(const XCSF *xcsf, const CL *c)
{
    (void)c;
    return xcsf->num_x_vars;
}

size_t cond_rectangle_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    size_t s = 0;
    const COND_RECTANGLE *cond = c->cond;
    s += fwrite(cond->center, sizeof(double), xcsf->num_x_vars, fp);
    s += fwrite(cond->spread, sizeof(double), xcsf->num_x_vars, fp);
    return s;
}

size_t cond_rectangle_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    size_t s = 0;
    COND_RECTANGLE *new = malloc(sizeof(COND_RECTANGLE));
    new->center = malloc(sizeof(double) * xcsf->num_x_vars);
    new->spread = malloc(sizeof(double) * xcsf->num_x_vars);
    s += fread(new->center, sizeof(double), xcsf->num_x_vars, fp);
    s += fread(new->spread, sizeof(double), xcsf->num_x_vars, fp);
    c->cond = new;
    return s;
}
