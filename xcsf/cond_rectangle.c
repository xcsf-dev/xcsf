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
#include "sam.h"
#include "cl.h"
#include "condition.h"
#include "cond_rectangle.h"

#define N_MU 1 //!< Number of hyperrectangle mutation rates

static double cond_rectangle_dist(const XCSF *xcsf, const CL *c, const double *x);

void cond_rectangle_init(const XCSF *xcsf, CL *c)
{
    COND_RECTANGLE *new = malloc(sizeof(COND_RECTANGLE));
    new->center = malloc(xcsf->x_dim * sizeof(double));
    new->spread = malloc(xcsf->x_dim * sizeof(double));
    for(int i = 0; i < xcsf->x_dim; i++) {
        new->center[i] = rand_uniform(xcsf->COND_MIN, xcsf->COND_MAX);
        new->spread[i] = rand_uniform(xcsf->COND_SMIN, fabs(xcsf->COND_MAX - xcsf->COND_MIN));
    }
    new->mu = malloc(N_MU * sizeof(double));
    sam_init(xcsf, new->mu, N_MU);
    c->cond = new;
}

void cond_rectangle_free(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const COND_RECTANGLE *cond = c->cond;
    free(cond->center);
    free(cond->spread);
    free(cond->mu);
    free(c->cond);
}

void cond_rectangle_copy(const XCSF *xcsf, CL *dest, const CL *src)
{
    COND_RECTANGLE *new = malloc(sizeof(COND_RECTANGLE));
    const COND_RECTANGLE *src_cond = src->cond;
    new->center = malloc(xcsf->x_dim * sizeof(double));
    new->spread = malloc(xcsf->x_dim * sizeof(double));
    new->mu = malloc(N_MU * sizeof(double));
    memcpy(new->center, src_cond->center, xcsf->x_dim * sizeof(double));
    memcpy(new->spread, src_cond->spread, xcsf->x_dim * sizeof(double));
    memcpy(new->mu, src_cond->mu, N_MU * sizeof(double));
    dest->cond = new;
}

void cond_rectangle_cover(const XCSF *xcsf, const CL *c, const double *x)
{
    const COND_RECTANGLE *cond = c->cond;
    for(int i = 0; i < xcsf->x_dim; i++) {
        cond->center[i] = x[i];
        cond->spread[i] = rand_uniform(xcsf->COND_SMIN, fabs(xcsf->COND_MAX - xcsf->COND_MIN));
    }
}

void cond_rectangle_update(const XCSF *xcsf, const CL *c, const double *x,
                           const double *y)
{
    (void)y;
    if(xcsf->COND_ETA > 0) {
        const COND_RECTANGLE *cond = c->cond;
        for(int i = 0; i < xcsf->x_dim; i++) {
            cond->center[i] += xcsf->COND_ETA * (x[i] - cond->center[i]);
        }
    }
}

_Bool cond_rectangle_match(const XCSF *xcsf, const CL *c, const double *x)
{
    if(cond_rectangle_dist(xcsf, c, x) < 1) {
        return true;
    }
    return false;
}

static double cond_rectangle_dist(const XCSF *xcsf, const CL *c, const double *x)
{
    const COND_RECTANGLE *cond = c->cond;
    double dist = 0;
    for(int i = 0; i < xcsf->x_dim; i++) {
        double d = fabs((x[i] - cond->center[i]) / cond->spread[i]);
        if(d > dist) {
            dist = d; // max distance
        }
    }
    return dist;
}

_Bool cond_rectangle_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    const COND_RECTANGLE *cond1 = c1->cond;
    const COND_RECTANGLE *cond2 = c2->cond;
    _Bool changed = false;
    if(rand_uniform(0, 1) < xcsf->P_CROSSOVER) {
        for(int i = 0; i < xcsf->x_dim; i++) {
            if(rand_uniform(0, 1) < 0.5) {
                double tmp = cond1->center[i];
                cond1->center[i] = cond2->center[i];
                cond2->center[i] = tmp;
                changed = true;
            }
            if(rand_uniform(0, 1) < 0.5) {
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
    sam_adapt(xcsf, cond->mu, N_MU);
    _Bool changed = false;
    for(int i = 0; i < xcsf->x_dim; i++) {
        double orig = cond->center[i];
        cond->center[i] += rand_normal(0, cond->mu[0]);
        cond->center[i] = clamp(xcsf->COND_MIN, xcsf->COND_MAX, cond->center[i]);
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

_Bool cond_rectangle_general(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    const COND_RECTANGLE *cond1 = c1->cond;
    const COND_RECTANGLE *cond2 = c2->cond;
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

void cond_rectangle_print(const XCSF *xcsf, const CL *c)
{
    const COND_RECTANGLE *cond = c->cond;
    printf("rectangle:");
    for(int i = 0; i < xcsf->x_dim; i++) {
        printf(" (c=%5f, ", cond->center[i]);
        printf("s=%5f)", cond->spread[i]);
    }
    printf("\n");
}

int cond_rectangle_size(const XCSF *xcsf, const CL *c)
{
    (void)c;
    return xcsf->x_dim;
}

size_t cond_rectangle_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    size_t s = 0;
    const COND_RECTANGLE *cond = c->cond;
    s += fwrite(cond->center, sizeof(double), xcsf->x_dim, fp);
    s += fwrite(cond->spread, sizeof(double), xcsf->x_dim, fp);
    s += fwrite(cond->mu, sizeof(double), N_MU, fp);
    return s;
}

size_t cond_rectangle_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    size_t s = 0;
    COND_RECTANGLE *new = malloc(sizeof(COND_RECTANGLE));
    new->center = malloc(xcsf->x_dim * sizeof(double));
    new->spread = malloc(xcsf->x_dim * sizeof(double));
    new->mu = malloc(N_MU * sizeof(double));
    s += fread(new->center, sizeof(double), xcsf->x_dim, fp);
    s += fread(new->spread, sizeof(double), xcsf->x_dim, fp);
    s += fread(new->mu, sizeof(double), N_MU, fp);
    c->cond = new;
    return s;
}
