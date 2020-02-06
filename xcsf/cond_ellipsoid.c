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
 * @file cond_ellipsoid.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Hyperellipsoid condition functions.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "sam.h"
#include "cl.h"
#include "condition.h"
#include "cond_ellipsoid.h"

#define N_MU 1 //!< Number of hyperellipsoid mutation rates

/**
 * @brief Hyperellipsoid condition data structure.
 */ 
typedef struct COND_ELLIPSOID {
    double *center; //!< Centers
    double *spread; //!< Spreads
    double mu[N_MU]; //!< Mutation rates
} COND_ELLIPSOID;

static double cond_ellipsoid_dist(const XCSF *xcsf, const CL *c, const double *x);

void cond_ellipsoid_init(const XCSF *xcsf, CL *c)
{
    COND_ELLIPSOID *new = malloc(sizeof(COND_ELLIPSOID));
    new->center = malloc(sizeof(double) * xcsf->x_dim);
    new->spread = malloc(sizeof(double) * xcsf->x_dim);
    for(int i = 0; i < xcsf->x_dim; i++) {
        new->center[i] = rand_uniform(xcsf->COND_MIN, xcsf->COND_MAX);
        new->spread[i] = rand_uniform(xcsf->COND_SMIN, fabs(xcsf->COND_MAX - xcsf->COND_MIN));
    }
    sam_init(xcsf, new->mu, N_MU);
    c->cond = new;
}

void cond_ellipsoid_free(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const COND_ELLIPSOID *cond = c->cond;
    free(cond->center);
    free(cond->spread);
    free(c->cond);
}

void cond_ellipsoid_copy(const XCSF *xcsf, CL *to, const CL *from)
{
    COND_ELLIPSOID *new = malloc(sizeof(COND_ELLIPSOID));
    const COND_ELLIPSOID *from_cond = from->cond;
    new->center = malloc(sizeof(double) * xcsf->x_dim);
    new->spread = malloc(sizeof(double) * xcsf->x_dim);
    memcpy(new->center, from_cond->center, sizeof(double) * xcsf->x_dim);
    memcpy(new->spread, from_cond->spread, sizeof(double) * xcsf->x_dim);
    memcpy(new->mu, from_cond->mu, sizeof(double) * N_MU);
    to->cond = new;
}                             

void cond_ellipsoid_cover(const XCSF *xcsf, const CL *c, const double *x)
{
    const COND_ELLIPSOID *cond = c->cond;
    for(int i = 0; i < xcsf->x_dim; i++) {
        cond->center[i] = x[i];
        cond->spread[i] = rand_uniform(xcsf->COND_SMIN, fabs(xcsf->COND_MAX - xcsf->COND_MIN));
    }
}

void cond_ellipsoid_update(const XCSF *xcsf, const CL *c, const double *x, const double *y)
{
    (void)y;
    if(xcsf->COND_ETA > 0) {
        const COND_ELLIPSOID *cond = c->cond;
        for(int i = 0; i < xcsf->x_dim; i++) {
            cond->center[i] += xcsf->COND_ETA * (x[i] - cond->center[i]);
        }
    }
}

_Bool cond_ellipsoid_match(const XCSF *xcsf, const CL *c, const double *x)
{
    if(cond_ellipsoid_dist(xcsf, c, x) < 1) {
        return true;
    }
    return false;
}

static double cond_ellipsoid_dist(const XCSF *xcsf, const CL *c, const double *x)
{
    const COND_ELLIPSOID *cond = c->cond;
    double dist = 0;
    for(int i = 0; i < xcsf->x_dim; i++) {
        double d = (x[i] - cond->center[i]) / cond->spread[i];
        dist += d*d; // squared distance
    }
    return dist;
}

_Bool cond_ellipsoid_crossover(const XCSF *xcsf, const CL *c1, const CL *c2) 
{
    const COND_ELLIPSOID *cond1 = c1->cond;
    const COND_ELLIPSOID *cond2 = c2->cond;
    _Bool changed = false;
    if(rand_uniform(0,1) < xcsf->P_CROSSOVER) {
        for(int i = 0; i < xcsf->x_dim; i++) {
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

_Bool cond_ellipsoid_mutate(const XCSF *xcsf, const CL *c)
{
    COND_ELLIPSOID *cond = c->cond;
    sam_adapt(xcsf, cond->mu, N_MU);
    _Bool changed = false;
    for(int i = 0; i < xcsf->x_dim; i++) {
        double orig = cond->center[i];
        cond->center[i] += rand_normal(0, cond->mu[0]);
        cond->center[i] = constrain(xcsf->COND_MIN, xcsf->COND_MAX, cond->center[i]);
        if(orig != cond->center[i]) {
            changed = true;
        }
        orig = cond->spread[i];
        cond->spread[i] += rand_normal(0, cond->mu[0]);
        if(orig != cond->spread[i]) {
            changed = true;
        }
    }
    return changed;   
}

_Bool cond_ellipsoid_general(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    const COND_ELLIPSOID *cond1 = c1->cond;
    const COND_ELLIPSOID *cond2 = c2->cond;
    for(int i = 0; i < xcsf->x_dim; i++) {
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

void cond_ellipsoid_print(const XCSF *xcsf, const CL *c)
{
    const COND_ELLIPSOID *cond = c->cond;
    printf("ellipsoid:");
    for(int i = 0; i < xcsf->x_dim; i++) {
        printf(" (%5f, ", cond->center[i]);
        printf("%5f)", cond->spread[i]);
    }
    printf("\n");
}

int cond_ellipsoid_size(const XCSF *xcsf, const CL *c)
{
    (void)c;
    return xcsf->x_dim;
}

size_t cond_ellipsoid_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    size_t s = 0;
    const COND_ELLIPSOID *cond = c->cond;
    s += fwrite(cond->center, sizeof(double), xcsf->x_dim, fp);
    s += fwrite(cond->spread, sizeof(double), xcsf->x_dim, fp);
    s += fwrite(cond->mu, sizeof(double), N_MU, fp);
    return s;
}

size_t cond_ellipsoid_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    size_t s = 0;
    COND_ELLIPSOID *new = malloc(sizeof(COND_ELLIPSOID));
    new->center = malloc(sizeof(double) * xcsf->x_dim);
    new->spread = malloc(sizeof(double) * xcsf->x_dim);
    s += fread(new->center, sizeof(double), xcsf->x_dim, fp);
    s += fread(new->spread, sizeof(double), xcsf->x_dim, fp);
    s += fread(new->mu, sizeof(double), N_MU, fp);
    c->cond = new;
    return s;
}
